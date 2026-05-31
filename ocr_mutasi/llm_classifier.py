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
statement (BCA or BRI) into one of four categories.

You see ONLY ONE statement's credits at a time, so you cannot verify monthly \
recurrence. Rely on explicit labels and amount/source plausibility.

- "Gaji"      — regular salary / payroll deposit. Strongest signal: an \
explicit payroll-system or salary label in the description. Common Indonesian \
labels include: GAJI, PAYROLL, SALARY, KR OTOMATIS GAJI, "TRSF GAJI", \
SAP-DD (SAP Direct Deposit, used by many corporates for payroll), \
PAYROLL-DEPOSIT, SALARY-CRDT. **Any row whose description clearly contains \
one of these labels should be classified Gaji** even when other context is \
missing — these labels are payroll-system identifiers, not generic terms.
- "Tunjangan" — allowance. Labels include THR, ECUTI (extra cuti / leave \
allowance), TUNJANGAN, ALLOWANCE; usually smaller than salary or tied to \
specific months (Lebaran for THR, leave periods for ECUTI).
- "Bonus"    — irregular bonus / commission / performance pay. Labels: BONUS, \
BONUS_INTERIM, BONUS_POOL, KOMISI, INSENTIF, COMMISSION. Amount varies, often \
larger than salary, timing irregular.
- "Lainnya"  — anything else: peer-to-peer transfers (Transfer Dari …, BIF \
TRANSFER DR from a person's name), refunds, interest, sale proceeds, \
self-transfers, reimbursements. Use this when the description is generic and \
no payroll/allowance/bonus label is present.

The user sends a JSON array of credit rows. Return strict JSON matching the \
schema. For each row include a short reason (≤20 words). Prefer Lainnya only \
when the row is genuinely generic — do NOT downgrade an explicit payroll-system \
label to Lainnya just because you cannot see cross-month recurrence."""


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
                        "category": {"type": "string", "enum": ["Gaji", "Tunjangan", "Bonus", "Lainnya"]},
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
statements. You are seeing credits from MULTIPLE monthly statements at once, \
so you can detect recurring patterns across months — which is the strongest \
signal for distinguishing salary from one-off transfers.

Categories:
- "Gaji"      — regular monthly payroll. STRONGEST SIGNAL: the same (or very \
similar) amount appears in MULTIPLE different months at roughly the same \
day-of-month, from the same source/system. Common Indonesian payroll-system \
labels: GAJI, PAYROLL, SALARY, KR OTOMATIS, SAP-DD, "TRSF GAJI". \
**Recurrence beats keywords:** if you see a deposit that recurs monthly with \
similar amount and timing, classify it Gaji even if the description is opaque.
- "Tunjangan" — periodic allowance (THR, transport, kesehatan, pulsa, leave \
allowance like ECUTI). Often labeled TUNJANGAN/THR/ECUTI/ALLOWANCE; smaller \
than salary; appears once or twice per year or in specific months.
- "Bonus"    — irregular bonus/commission/performance pay. Often labeled \
BONUS/KOMISI/INSENTIF. Amount varies; non-recurring; usually larger than salary.
- "Lainnya"  — anything else: peer-to-peer transfers, refunds, interest, \
sale proceeds, reimbursements, self-transfers. Use this when the row clearly \
does not match Gaji/Tunjangan/Bonus.

The user sends a JSON array. Each item has: id (int), source_file (the PDF \
this row came from), tanggal (ISO date), amount, keterangan (description). \
Return strict JSON with one classification per id. Reason ≤25 words. \
Be conservative on Gaji UNLESS recurrence is clear; prefer Lainnya when \
uncertain."""


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
