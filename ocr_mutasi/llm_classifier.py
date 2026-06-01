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
recurrence. Rely on explicit labels and amount/source plausibility.

- "Gaji"     — fixed monthly salary / payroll deposit. The amount is the \
same every month and arrives on/near the same day. Common payroll-system \
labels in Indonesian statements: GAJI, PAYROLL, SALARY, KR OTOMATIS GAJI, \
"TRSF GAJI", SAP-DD (SAP Direct Deposit — widely used for corporate payroll), \
PAYROLL-DEPOSIT, SALARY-CRDT. A row whose description clearly carries one of \
these labels should be classified Gaji even with no other context.
- "THR"      — Tunjangan Hari Raya, a religious-holiday allowance paid one \
or two times a year (around Idul Fitri / Lebaran, or Christmas). Labels: \
THR, THR_Islam, THR_Idulfitri, THR_Lebaran, HARI RAYA, TUNJANGAN HARI RAYA. \
Typically larger than monthly Gaji.
- "Bonus"    — annual extra payment, usually paid once per year. Labels: \
BONUS, BONUS_TAHUNAN, BONUS_POOL, BONUS_YEARLY, ANNUAL_BONUS, year-end-bonus \
descriptors. Structured / scheduled rather than performance-triggered.
- "Insentif" — performance-based extra payment whose amount varies with \
individual or team results. Labels: INSENTIF, INCENTIVE, KOMISI, COMMISSION, \
BONUS_INTERIM (interim performance), PERFORMANCE_BONUS, COMMISSION_PAY. \
Often quarterly or tied to sales periods.
- "Lainnya"  — anything else: peer-to-peer transfers (Transfer Dari …, BIF \
TRANSFER DR from a person's name), refunds, interest, sale proceeds, \
self-transfers, reimbursements, leave allowances such as ECUTI, and generic \
monthly tunjangan (transport / health / pulsa) that don't match the four \
specific categories above.

The user sends a JSON array of credit rows. Return strict JSON matching the \
schema. For each row include a short reason (≤25 words). Prefer Lainnya only \
when the row is genuinely generic — do NOT downgrade an explicit \
Gaji/THR/Bonus/Insentif label to Lainnya just because you cannot see \
cross-month recurrence."""


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
statements at once, so you can detect recurring patterns across months — \
which is the strongest signal for distinguishing salary from one-off transfers.

Five categories:

- "Gaji"     — fixed monthly salary / payroll deposit. STRONGEST SIGNAL: \
the same (or very similar) amount appears in MULTIPLE different months at \
roughly the same day-of-month, from the same source/system. Common payroll \
labels in Indonesian statements: GAJI, PAYROLL, SALARY, KR OTOMATIS, SAP-DD, \
"TRSF GAJI", PAYROLL-DEPOSIT. **Recurrence beats keywords:** a deposit that \
recurs monthly with similar amount and timing IS Gaji even if the description \
is opaque.
- "THR"      — Tunjangan Hari Raya, religious-holiday allowance paid one or \
two times per year (Idul Fitri / Lebaran, sometimes Christmas). Labels: THR, \
THR_Islam, THR_Idulfitri, THR_Lebaran, HARI RAYA, TUNJANGAN HARI RAYA. \
Typically larger than monthly Gaji and lands in a specific month each year.
- "Bonus"    — annual extra payment, usually once per year (year-end or \
similar). Labels: BONUS, BONUS_TAHUNAN, BONUS_POOL, BONUS_YEARLY, \
ANNUAL_BONUS. Scheduled / structured rather than triggered by performance.
- "Insentif" — performance-based extra payment, variable amount based on \
individual or team performance. Labels: INSENTIF, INCENTIVE, KOMISI, \
COMMISSION, BONUS_INTERIM (interim/performance), PERFORMANCE_BONUS, \
COMMISSION_PAY. Often quarterly or tied to sales periods.
- "Lainnya"  — anything else: peer-to-peer transfers, refunds, interest, \
sale proceeds, reimbursements, self-transfers, leave allowances (ECUTI), \
generic monthly tunjangan (transport / health / pulsa) that don't match the \
four specific categories above.

Disambiguation tips:
- BONUS_POOL / BONUS_TAHUNAN → Bonus (annual / structured).
- BONUS_INTERIM / COMMISSION → Insentif (performance / variable).
- THR_* / HARI RAYA → THR (religious-holiday timing).
- ECUTI / TUNJANGAN TRANSPORT / monthly perks → Lainnya (don't fit the \
specific four).

The user sends a JSON array. Each item has: id (int), source_file (the PDF \
this row came from), tanggal (ISO date), amount, keterangan (description). \
Return strict JSON with one classification per id. Reason ≤25 words. \
Be conservative on Gaji UNLESS recurrence is clear; prefer Lainnya when \
truly uncertain."""


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
