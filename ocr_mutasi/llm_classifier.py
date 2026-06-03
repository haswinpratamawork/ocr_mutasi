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
statement (BCA, BRI, Mandiri, Permata, or Sinarmas) into one of five categories.

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

4. **Gaji — four flavours.** Flavours (a) and (c) REQUIRE a CORPORATE \
SENDER (e.g. `PT <X>`, `<X> PT`, `<X> INDO`, `<X> JAKARTA`, `<X> BSD`, \
`<X> ALAM SUTERA`, `<X> TBK`, `CV <X>`, `KASTARA <X>`, etc. — a registered- \
company-style name, NOT a personal name like `BUDI SANTOSO` or `DRG.JOKO \
HARTONO`). Flavours (b) and (d) are bank-product labels self-sufficient on \
their own.

   **Bank names are NOT corporate senders.** Strings like `PT. BANK <X>` \
(e.g. `PT. BANK JASA JAKARTA`, `PT. BANK CENTRAL ASIA`, `PT. BANK MANDIRI`, \
`PT. BANK NEGARA INDONESIA`, `PT. BANK PERMATA`, `PT. BANK SINARMAS`, \
`PT. BANK RAKYAT INDONESIA`, `PT. BANK CIMB NIAGA`, `PT. BANK DANAMON`, \
`PT. BANK MAYBANK INDONESIA`, `PT. BANK OCBC NISP`, `PT. BANK UOB INDONESIA`, \
`PT. BANK BTPN`, `PT. BANK BCA SYARIAH`, etc.) are the COUNTERPARTY'S BANK \
(routing info, not the sender). For BIFAST / BI Fast / RTGS / LLG-style \
transfers, the actual sender is the person/company NAME that appears \
alongside the bank — typically AFTER an account number, or before/after \
the bank in the same row. If that sender NAME is a personal first name \
(`SITI`, `BUDI`, `JONI`, `JOKO`, ...) or two-word personal name → \
Lainnya per rule 5. A bank name with `PT.` prefix is NOT a payroll source.

   (a) **Employer-facing payroll labels** + corporate sender: `GAJI`, \
`PAYROLL`, `SALARY`, `TRSF GAJI`, `PAYROLL-DEPOSIT`, `SALARY-CRDT`.
   (b) **Bank bulk-payroll product labels — self-sufficient** (the label \
alone is the signal; no corporate sender required because these channels \
don't always expose the employer name): `SAP-DD` (SAP Direct Deposit), \
`KR OTOMATIS` (BCA auto-credit, when NOT accompanied by an `LLG` label — \
rule 3 catches the LLG case first), `SMEMFTS` (BCA SME Mass Funds Transfer \
Service — the primary salary channel).
   (c) **Professional-fee / honorarium labels** + corporate sender — \
payment FOR work done, where the description names the kind of work and \
the sender is a company: `FEE DOKTER`, `FEE DRG` (Dokter Gigi), \
`FEE NOTARIS`, `FEE INSINYUR`, `FEE KONSULTAN`, `FEE PENGACARA`, \
`FEE [profession]`, `HONOR`, `HONORARIUM`, `JASA <kind-of-service>`, \
`RETAINER`. `JASA` must be followed by a service-kind word (e.g. `JASA \
KONSULTAN`, `JASA DESAIN`, `JASA HUKUM`); `BANK JASA <city>` is a bank \
name, NOT a `JASA <name>` fee. Example matches: `TRSF E-BANKING CR <ref> \
| FEE DOKTER | PT KLINIK CONTOH JAKARTA` → **Gaji**, `TRSF E-BANKING CR \
<ref> | FEE DRG JOKO | KLINIK CONTOH BSD` → **Gaji**.
   (d) **Sinarmas payroll auto-channel — self-sufficient**: any description \
containing `AUTO TRANSFER CREDIT` is the Sinarmas Tabungan payroll auto- \
deposit channel and classifies as Gaji on the label alone, with NO \
corporate sender required (the Sinarmas channel does not expose the \
employer name in the row; only a branch code like `BSD` accompanies it). \
Example: `BSD | Auto Transfer Credit` → **Gaji**.

   **Hard exclusion for rule 4 (overrides any match above):** if the \
description contains `CASHBACK`, `REFUND`, `REIMBURSE`, `REIMBURSEMENT`, \
`BUNGA` (bank interest), `TAX REFUND`, or `PROMO`, treat as Lainnya — these \
are merchant/bank disbursements, not employer payments, even if the apparent \
"sender" happens to look corporate (e.g. `KR OTOMATIS TRF KOLEKTIF \| \
CASHBACK QRIS BCA \| DI MERCHANT XYZ` is Lainnya, not Gaji).

5. **Otherwise → "Lainnya"** — peer-to-peer transfers from a person's name \
(`Transfer Dari <name>`, `BIF TRANSFER DR <name>`, `BI Fast Payment Cr | \
PT. BANK <X> | <acct-no> | SITI`, or any clearly-personal sender like \
`BUDI SANTOSO`, `DRG.JOKO HARTONO`, `JONI WIJAYA`, `SITI`), refunds, \
interest (`BUNGA`, `Credit Interest`), sale proceeds, self-transfers, \
reimbursements, debt repayments (Indonesian `hutang`), anything that \
lacks the labels in rules 1–4. **In particular: a `BI Fast Payment Cr` / \
`BIF TRANSFER` whose only "corporate-looking" string is a bank name \
(`PT. BANK …`) and whose sender NAME is a personal first name is \
ALWAYS Lainnya — the bank is routing, the person is the sender.**

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
statements (BCA, BRI, Mandiri, Permata, Sinarmas). You see credits from \
MULTIPLE monthly statements at once, so you can detect recurring patterns \
across months. \
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

4. **Gaji — four flavours.** Flavours (a) and (c) require a CORPORATE \
SENDER (`PT <X>`, `<X> PT`, `<X> INDO`, `<X> JAKARTA`, `<X> BSD`, `<X> ALAM \
SUTERA`, `<X> TBK`, `CV <X>`, `KASTARA <X>`, etc. — a registered-company- \
style name, NOT a personal name). Flavours (b) and (d) are bank-product \
labels self-sufficient on their own.

   **Bank names are NOT corporate senders.** Strings like `PT. BANK <X>` \
(`PT. BANK JASA JAKARTA`, `PT. BANK CENTRAL ASIA`, `PT. BANK MANDIRI`, \
`PT. BANK NEGARA INDONESIA`, `PT. BANK PERMATA`, `PT. BANK SINARMAS`, \
`PT. BANK RAKYAT INDONESIA`, `PT. BANK CIMB NIAGA`, `PT. BANK DANAMON`, \
`PT. BANK MAYBANK INDONESIA`, `PT. BANK OCBC NISP`, `PT. BANK UOB \
INDONESIA`, `PT. BANK BTPN`, etc.) are the COUNTERPARTY'S BANK (routing \
info, not the sender). For BIFAST / BI Fast / RTGS / LLG-style transfers, \
the actual sender is the person/company NAME alongside the bank — \
typically after an account number. If that sender NAME is a personal first \
name (`SITI`, `BUDI`, `JOKO`, ...) or a two-word personal name → Lainnya \
per rule 6. A bank name with `PT.` prefix is NOT a payroll source.

   (a) **Employer-facing payroll labels** + corporate sender: `GAJI`, \
`PAYROLL`, `SALARY`, `TRSF GAJI`, `PAYROLL-DEPOSIT`, `SALARY-CRDT`.
   (b) **Bank bulk-payroll product labels — self-sufficient** (the label \
alone is the signal; no corporate sender required): `SAP-DD`, `KR OTOMATIS` \
(when no `LLG` is present — rule 3 catches the LLG case first), `SMEMFTS` \
(BCA SME Mass Funds Transfer Service — the primary salary channel).
   (c) **Professional-fee / honorarium labels** + corporate sender — \
payment FOR work done, where the description names the kind of work and \
the sender is a company: `FEE DOKTER`, `FEE DRG`, `FEE NOTARIS`, \
`FEE INSINYUR`, `FEE KONSULTAN`, `FEE PENGACARA`, `FEE [profession]`, \
`HONOR`, `HONORARIUM`, `JASA <kind-of-service>`, `RETAINER`. `JASA` must \
be followed by a service-kind word (`JASA KONSULTAN`, `JASA DESAIN`, \
`JASA HUKUM`); `BANK JASA <city>` is a bank name, NOT a `JASA <name>` \
fee. Example matches: `TRSF E-BANKING CR <ref> \| FEE DOKTER \| PT OSG \
JAKARTA TIM` → **Gaji**; `TRSF E-BANKING CR <ref> \| FEE DRG JOKO \| \
KLINIK CONTOH BSD` → **Gaji**.
   (d) **Sinarmas payroll auto-channel — self-sufficient**: any description \
containing `AUTO TRANSFER CREDIT` is the Sinarmas Tabungan payroll auto- \
deposit channel and classifies as Gaji on the label alone, with NO corporate \
sender required (the Sinarmas channel does not expose the employer name in \
the row — only a branch code like `BSD` accompanies it). Example: \
`BSD | Auto Transfer Credit` → **Gaji**.

   **Hard exclusion for rule 4 (overrides any match above):** if the \
description contains `CASHBACK`, `REFUND`, `REIMBURSE`, `REIMBURSEMENT`, \
`BUNGA` (bank interest), `TAX REFUND`, or `PROMO`, classify as Lainnya \
regardless of any other keyword — these are merchant/bank disbursements, \
not employer payments, even when the apparent "sender" looks corporate.

   When one of these labels appears together with a corporate sender name \
(e.g. `PT TUV RHEINLAND`, `TUV RHEINLAND INDO`, or any `PT <X>` / `<X> INDO`), \
it's almost certainly Gaji — DO NOT downgrade it to Lainnya.

5. **No label match? Use cross-month recurrence.** The strongest cross-month \
salary signal is **the same CORPORATE SENDER appearing across multiple \
months**, not amount equality. Real salaries vary monthly due to overtime, \
deductions, prorated months, raises, or bundled THR/bonus. A credit whose \
description names the SAME corporate sender (`PT <X>`, `<X> INDO`, `<X> PT`, \
`KASTARA <X>`, etc.) and that you can see in ≥ 2 different months → \
"Gaji", even if amounts range widely (e.g. 400K, 11M, 47M; or 5.9M, 6.3M, \
5.8M for monthly professional fees). Day-of-month consistency is a weaker \
secondary hint. The same hard exclusion as rule 4 applies: `CASHBACK`, \
`REFUND`, `BUNGA`, `PROMO` → Lainnya, even with recurrence. **Bank names \
are not corporate senders** — recurrence across months of `PT. BANK <X>` \
strings is just the counterparty repeatedly using the same bank.

6. **Otherwise → "Lainnya"** — peer-to-peer transfers (`Transfer Dari \
<name>`, `BIF TRANSFER DR <name>`, `BI Fast Payment Cr | PT. BANK <X> | \
<acct-no> | SITI`-style rows), refunds, interest (`BUNGA`, `Credit \
Interest`), sale proceeds, reimbursements, self-transfers, debt \
repayments (Indonesian `hutang`), anything without any of the labels \
above. **In particular: a `BI Fast Payment Cr` / `BIF TRANSFER` whose \
only "corporate-looking" string is a bank name (`PT. BANK …`) and whose \
sender NAME is a personal first name is ALWAYS Lainnya — the bank is \
routing, the person is the sender.**

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
