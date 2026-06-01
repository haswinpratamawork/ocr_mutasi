"""Classify credit transactions into Gaji / Tunjangan / Bonus / Lainnya via Azure OpenAI.

Two entry points:

* `classify_credits(credits)` — used by single-PDF flow. Sees only one month.
* `classify_credits_batch(credits_with_source)` — used by /extract-batch.
  Receives credits from every uploaded PDF in a single request so the model
  can exploit cross-month recurrence (the strongest `Gaji` signal). Each
  credit carries its source filename and date so the model can reason about
  same-day-of-month, same-amount, same-source patterns.

Both paths constrain the response with a JSON schema so we can deserialize
straight into Pydantic models with zero string parsing.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from openai import APIError, APITimeoutError, AzureOpenAI

from .config import get_settings
from .models import ClassifiedCredit, Transaction

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You classify INCOMING credit rows from an Indonesian bank \
statement (BCA, BRI, or Mandiri) into one of five categories.

You see ONLY ONE statement's credits at a time, so you cannot verify monthly \
recurrence. The classification is mostly driven by an explicit label in the \
description (Indonesian corporate payroll systems use very specific tokens). \
Match labels first, fall back to amount/source heuristics second.

## Label-first decision rules (apply IN ORDER, first match wins)

1. **Any description containing `BONUS_…` or `BONUS ` → "Bonus"** — \
INCLUDING `BONUS_INTERIM` (interim bonus), `BONUS_POOL`, `BONUS_TAHUNAN`, \
`BONUS_YEARLY`, `ANNUAL_BONUS`, year-end bonus descriptors. The `BONUS_` \
naming is the company's own bonus-program prefix; whether it's annual or \
interim doesn't matter — it's still Bonus. **DO NOT classify a `BONUS_*` \
label as Insentif.**

2. **Any description containing `THR`, `HARI RAYA`, or `TUNJANGAN HARI RAYA` \
→ "THR"** — Tunjangan Hari Raya, religious-holiday allowance. Examples: \
`THR_Islam`, `THR_Idulfitri`, `THR_Lebaran`. Usually larger than Gaji. \
(This rule must come before rule 3 so `TUNJANGAN HARI RAYA` is routed to \
THR rather than the generic-tunjangan rule below.)

3. **Any description containing one of the following → "Insentif"** — \
performance- or work-related extra payments:
   - explicit incentive labels: `ECUTI` (extra cuti / extra-leave payout), \
`INSENTIF`, `INCENTIVE`, `KOMISI`, `COMMISSION`, `PERFORMANCE_BONUS`, \
`COMMISSION_PAY`;
   - work-related `TUNJANGAN <kind>` allowances (i.e. any TUNJANGAN that \
isn't TUNJANGAN HARI RAYA): `TUNJANGAN TRANSPORT` / `TRANSPORTASI`, \
`TUNJANGAN MAKAN` / `UANG MAKAN` / `MEAL_ALLOWANCE`, `TUNJANGAN PULSA` / \
`PHONE_ALLOWANCE`, `TUNJANGAN KELUAR KOTA` / `DINAS LUAR KOTA` / \
`TRAVEL_ALLOWANCE`, `TUNJANGAN KESEHATAN`, `TUNJANGAN ANAK`, \
`TUNJANGAN ISTRI`, and so on;
   - **Lalu Lintas Giro allowance channels** — any description containing \
`LLG-DEUTSCHE BANK` or starting with `LLG ` (BI bulk-clearing channel used \
for allowance disbursement, distinct from the main payroll channel). A row \
like `KR OTOMATIS LLG-DEUTSCHE BANK | PT TUV RHEINLAND` is **Insentif** — \
this rule fires BEFORE rule 4's KR OTOMATIS match, so LLG always wins for \
mixed labels.

   These are work-tied perks paid alongside Gaji — they belong in Insentif, \
NOT in Lainnya.

4. **Any description containing one of the following payroll-disbursement \
labels → "Gaji"**:
   - employer-facing labels: `GAJI`, `PAYROLL`, `SALARY`, `TRSF GAJI`, \
`PAYROLL-DEPOSIT`, `SALARY-CRDT`;
   - Indonesian bank bulk-payroll product labels: `SAP-DD` (SAP Direct \
Deposit), `KR OTOMATIS` (BCA auto-credit, when NOT accompanied by an `LLG` \
label — rule 3 catches the LLG case first), `SMEMFTS` (BCA SME Mass Funds \
Transfer Service — the primary salary channel).

   When one of these labels appears together with a corporate sender name \
(e.g. `PT TUV RHEINLAND`, `TUV RHEINLAND INDO`, or any `PT <X>` / `<X> INDO`), \
it's almost certainly Gaji — DO NOT downgrade it to Lainnya.

5. **Otherwise → "Lainnya"** — peer-to-peer transfers from a person's name \
(`Transfer Dari <name>`, `BIF TRANSFER DR <name>`), refunds, interest, sale \
proceeds, self-transfers, reimbursements, anything that lacks the labels in \
rules 1–4.

## Output

Return strict JSON matching the schema. For each row include a short reason \
(≤25 words) that NAMES the label that drove your decision (e.g. \
"Contains BONUS_INTERIM label → Bonus per rule 1", "TUNJANGAN TRANSPORT \
label → Insentif per rule 3", or "SMEMFTS + PT TUV RHEINLAND sender → Gaji \
per rule 4"). Do NOT downgrade an explicit Gaji/THR/Bonus/Insentif label to \
Lainnya."""


_RESPONSE_SCHEMA = {
    "name": "credit_classifications",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "classifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "integer"},
                        "category": {"type": "string", "enum": ["Gaji", "THR", "Bonus", "Insentif", "Lainnya"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "reason": {"type": "string"},
                    },
                    "required": ["id", "category", "confidence", "reason"],
                },
            }
        },
        "required": ["classifications"],
    },
    "strict": True,
}


def classify_credits(credits: list[Transaction]) -> tuple[list[ClassifiedCredit], Optional[str]]:
    """Return (classified, error_message_or_None).

    On any LLM error we return the credits with `category=None` so the caller
    can still respond to the user with the extracted data plus a clear note.
    """
    if not credits:
        return [], None

    settings = get_settings()
    payload = [
        {"id": idx, "tanggal": tx.tanggal, "keterangan": tx.keterangan, "amount": tx.amount}
        for idx, tx in enumerate(credits)
    ]

    client = AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        timeout=settings.llm_request_timeout_s,
    )

    try:
        completion = client.chat.completions.create(
            model=settings.azure_openai_deployment,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({"credits": payload}, ensure_ascii=False)},
            ],
            response_format={"type": "json_schema", "json_schema": _RESPONSE_SCHEMA},
            temperature=0,
        )
        raw = completion.choices[0].message.content or "{}"
        decoded = json.loads(raw)
        by_id = {item["id"]: item for item in decoded.get("classifications", [])}
    except (APIError, APITimeoutError, json.JSONDecodeError, KeyError) as exc:
        logger.warning("LLM classification failed: %s", exc)
        return (
            [ClassifiedCredit(**tx.model_dump(), category=None, confidence=None, reason=None)
             for tx in credits],
            f"classifier error: {exc}",
        )

    out: list[ClassifiedCredit] = []
    for idx, tx in enumerate(credits):
        cls = by_id.get(idx)
        out.append(ClassifiedCredit(
            **tx.model_dump(),
            category=cls["category"] if cls else None,
            confidence=cls["confidence"] if cls else None,
            reason=cls["reason"] if cls else None,
        ))
    return out, None


# --------------------- batch (cross-PDF) classification --------------------

BATCH_SYSTEM_PROMPT = """You classify INCOMING credit rows from Indonesian bank \
statements (BCA, BRI, Mandiri). You see credits from MULTIPLE monthly \
statements at once, so you can detect recurring patterns across months. \
Classification is driven by an explicit label in the description first, with \
cross-month recurrence as a back-up signal for un-labelled rows.

## Label-first decision rules (apply IN ORDER, first match wins)

1. **Any description containing `BONUS_…` or `BONUS ` → "Bonus"** — \
INCLUDING `BONUS_INTERIM` (interim bonus), `BONUS_POOL`, `BONUS_TAHUNAN`, \
`BONUS_YEARLY`, `ANNUAL_BONUS`. The `BONUS_` prefix is the company's own \
bonus-program naming and ALL such rows are Bonus regardless of whether \
they're annual, interim, mid-year, or quarterly. **DO NOT split BONUS_INTERIM \
into Insentif.**

2. **Any description containing `THR`, `HARI RAYA`, or `TUNJANGAN HARI RAYA` \
→ "THR"** — Tunjangan Hari Raya, religious-holiday allowance. Examples: \
`THR_Islam`, `THR_Idulfitri`, `THR_Lebaran`. (This rule must come before \
rule 3 so `TUNJANGAN HARI RAYA` is routed to THR rather than the \
generic-tunjangan rule below.)

3. **Any description containing one of the following → "Insentif"** — \
performance- or work-related extra payments:
   - explicit incentive labels: `ECUTI` (extra cuti / extra-leave payout), \
`INSENTIF`, `INCENTIVE`, `KOMISI`, `COMMISSION`, `PERFORMANCE_BONUS`, \
`COMMISSION_PAY`;
   - work-related `TUNJANGAN <kind>` allowances (i.e. any TUNJANGAN that \
isn't TUNJANGAN HARI RAYA): `TUNJANGAN TRANSPORT` / `TRANSPORTASI`, \
`TUNJANGAN MAKAN` / `UANG MAKAN` / `MEAL_ALLOWANCE`, `TUNJANGAN PULSA` / \
`PHONE_ALLOWANCE`, `TUNJANGAN KELUAR KOTA` / `DINAS LUAR KOTA` / \
`TRAVEL_ALLOWANCE`, `TUNJANGAN KESEHATAN`, `TUNJANGAN ANAK`, \
`TUNJANGAN ISTRI`, and so on;
   - **Lalu Lintas Giro allowance channels** — any description containing \
`LLG-DEUTSCHE BANK` or starting with `LLG ` (BI bulk-clearing channel used \
for allowance disbursement, distinct from the main payroll channel). A row \
like `KR OTOMATIS LLG-DEUTSCHE BANK | PT TUV RHEINLAND` is **Insentif** — \
this rule fires BEFORE rule 4's KR OTOMATIS match, so LLG always wins for \
mixed labels.

   These are work-tied perks paid alongside Gaji — they belong in Insentif, \
NOT in Lainnya.

4. **Any description containing one of the following payroll-disbursement \
labels → "Gaji"**:
   - employer-facing labels: `GAJI`, `PAYROLL`, `SALARY`, `TRSF GAJI`, \
`PAYROLL-DEPOSIT`, `SALARY-CRDT`;
   - Indonesian bank bulk-payroll product labels: `SAP-DD` (SAP Direct \
Deposit), `KR OTOMATIS` (BCA auto-credit, when NOT accompanied by an `LLG` \
label — rule 3 catches the LLG case first), `SMEMFTS` (BCA SME Mass Funds \
Transfer Service — the primary salary channel).

   When one of these labels appears together with a corporate sender name \
(e.g. `PT TUV RHEINLAND`, `TUV RHEINLAND INDO`, or any `PT <X>` / `<X> INDO`), \
it's almost certainly Gaji — DO NOT downgrade it to Lainnya.

5. **No label match? Use cross-month recurrence.** The strongest cross-month \
salary signal is **the same SENDER appearing across multiple months**, not \
amount equality. Real salaries vary monthly due to overtime, deductions, \
prorated months, raises, or bundled THR/bonus. A credit whose description \
names the SAME corporate employer (e.g. `PT TUV RHEINLAND` / `TUV RHEINLAND \
INDO`) and that you can see in ≥ 3 different months → "Gaji", even if amounts \
range widely (e.g. 400K, 11M, 47M). Day-of-month consistency is a weaker \
secondary hint.

6. **Otherwise → "Lainnya"** — peer-to-peer transfers (`Transfer Dari <name>`, \
`BIF TRANSFER DR <name>`), refunds, interest, sale proceeds, reimbursements, \
self-transfers, anything without any of the labels above.

## Output

The user sends a JSON array. Each item has: id (int), source_file (PDF this \
row came from), tanggal (ISO date), amount, keterangan (description). Return \
strict JSON with one classification per id. Reason ≤25 words — name the label \
or pattern that drove the decision (e.g. "BONUS_INTERIM label → Bonus per \
rule 1", "TUNJANGAN TRANSPORT → Insentif per rule 3", or "Recurring monthly \
SAP-DD → Gaji per rule 5"). Do NOT downgrade an explicit \
Gaji/THR/Bonus/Insentif label to Lainnya."""


def classify_credits_batch(
    credits_with_source: list[tuple[str, Transaction]],
) -> tuple[list[ClassifiedCredit], Optional[str]]:
    """Cross-PDF classification.

    `credits_with_source` is a list of (source_filename, Transaction) pairs.
    Returns the same list re-typed as ClassifiedCredit (same order).
    """
    if not credits_with_source:
        return [], None

    settings = get_settings()
    payload = [
        {
            "id": idx,
            "source_file": src,
            "tanggal": tx.tanggal,
            "amount": tx.amount,
            "keterangan": tx.keterangan,
        }
        for idx, (src, tx) in enumerate(credits_with_source)
    ]

    client = AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        timeout=settings.llm_request_timeout_s * 2,  # batch is bigger; allow more
    )
    try:
        completion = client.chat.completions.create(
            model=settings.azure_openai_deployment,
            messages=[
                {"role": "system", "content": BATCH_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({"credits": payload}, ensure_ascii=False)},
            ],
            response_format={"type": "json_schema", "json_schema": _RESPONSE_SCHEMA},
            temperature=0,
        )
        raw = completion.choices[0].message.content or "{}"
        decoded = json.loads(raw)
        by_id = {item["id"]: item for item in decoded.get("classifications", [])}
    except (APIError, APITimeoutError, json.JSONDecodeError, KeyError) as exc:
        logger.warning("Batch LLM classification failed: %s", exc)
        return (
            [ClassifiedCredit(**tx.model_dump(), category=None, confidence=None, reason=None)
             for _, tx in credits_with_source],
            f"classifier error: {exc}",
        )

    out: list[ClassifiedCredit] = []
    for idx, (_, tx) in enumerate(credits_with_source):
        cls = by_id.get(idx)
        out.append(ClassifiedCredit(
            **tx.model_dump(),
            category=cls["category"] if cls else None,
            confidence=cls["confidence"] if cls else None,
            reason=cls["reason"] if cls else None,
        ))
    return out, None
