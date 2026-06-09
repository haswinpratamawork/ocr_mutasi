"""Scanned-statement fallback for ocr_mutasi.

Digital bank statements have a clean text layer and go through the fast,
deterministic bank parsers. When that fails — no embedded text / no known bank
layout, i.e. a scanned or photographed statement — we OCR the PDF via the
shared PaddleOCR service and ask the LLM to pull the transaction rows out of
the recognised text. This path is a fallback only.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from openai import APIError, APITimeoutError, AzureOpenAI

from .config import get_settings
from .models import AccountHeader, Transaction

logger = logging.getLogger(__name__)

# Cap the OCR text sent to the LLM to keep the request within token limits. A
# full year of statements can be long; the fallback is best-effort.
MAX_OCR_CHARS = 24_000

SYSTEM_PROMPT = """You extract EVERY transaction row from the OCR text of an \
Indonesian bank statement (mutasi rekening / rekening koran). The text is noisy \
(OCR artefacts, merged words, broken columns) — do your best to recover each row.

For every transaction return:
- "tanggal": the date as ISO `YYYY-MM-DD` (infer the year from the statement period if the row shows only day/month).
- "keterangan": the description / berita text for that row.
- "amount": the transaction amount as a positive number (no thousands separators, no currency symbol).
- "type": "CR" if the money came IN (credit / kredit / masuk), "DB" if it went OUT (debit / debet / keluar).
- "saldo": the running balance after the row as a number, or null if the row doesn't show one.

Only include real transaction rows. Skip headers, summaries, opening/closing \
balance lines, and totals. If the text contains no transactions, return an empty list."""

_RESPONSE_SCHEMA = {
    "name": "bank_transactions",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "transactions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "tanggal": {"type": "string"},
                        "keterangan": {"type": "string"},
                        "amount": {"type": "number"},
                        "type": {"type": "string", "enum": ["CR", "DB"]},
                        "saldo": {"type": ["number", "null"]},
                    },
                    "required": ["tanggal", "keterangan", "amount", "type", "saldo"],
                },
            }
        },
        "required": ["transactions"],
    },
    "strict": True,
}


def extract_transactions_via_llm(ocr_text: str) -> tuple[list[Transaction], Optional[str]]:
    """Return ``(transactions, error_or_None)`` parsed from OCR text by the LLM."""
    if not ocr_text.strip():
        return [], "no OCR text to extract from"

    settings = get_settings()
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
                {"role": "user", "content": ocr_text[:MAX_OCR_CHARS]},
            ],
            response_format={"type": "json_schema", "json_schema": _RESPONSE_SCHEMA},
            temperature=0,
        )
        raw = completion.choices[0].message.content or "{}"
        rows = json.loads(raw).get("transactions", [])
    except (APIError, APITimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("OCR LLM transaction extraction failed: %s", exc)
        return [], f"OCR LLM extraction error: {exc}"

    transactions: list[Transaction] = []
    for row in rows:
        try:
            saldo = row.get("saldo")
            transactions.append(
                Transaction(
                    tanggal=str(row["tanggal"]),
                    keterangan=str(row.get("keterangan", "")).strip(),
                    amount=float(row["amount"]),
                    type=row["type"],
                    saldo=float(saldo) if saldo is not None else None,
                    page=1,
                )
            )
        except (KeyError, ValueError, TypeError):
            continue  # drop malformed rows, keep the rest
    return transactions, None


def ocr_fallback_extract(
    pdf_bytes: bytes,
    password: str | None = None,
) -> tuple[AccountHeader, list[Transaction], list[str]]:
    """OCR a (scanned) statement and LLM-extract its transactions.

    Returns ``(account, transactions, warnings)``. Raises the imported
    ``PaddleOcrError`` family if the OCR service itself is unavailable, and
    ``ValueError`` if OCR/LLM produced nothing usable (the caller maps these to
    the existing UnsupportedBank/422 behaviour).
    """
    from ocr_common.paddle_ocr import extract_text_from_bytes

    ocr_text = extract_text_from_bytes(pdf_bytes, password=password)
    if not ocr_text.strip():
        raise ValueError("OCR produced no text from the document.")

    transactions, err = extract_transactions_via_llm(ocr_text)
    if not transactions:
        raise ValueError(
            "OCR + LLM fallback found no transactions"
            + (f" ({err})" if err else "") + "."
        )

    warnings = ["No known bank layout matched — extracted via OCR + LLM fallback."]
    if err:
        warnings.append(err)
    return AccountHeader(bank="OCR"), transactions, warnings
