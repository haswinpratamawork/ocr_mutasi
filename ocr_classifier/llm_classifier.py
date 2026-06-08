"""Classify a document's OCR text into one of the five Indonesian document
types (or `unknown`) via Azure OpenAI.

The decision is constrained with a JSON schema so the reply deserializes
straight into our enums with zero string parsing — the same technique
ocr_mutasi uses. On any LLM/parse error we degrade gracefully to
`unknown` with the error captured, so the service still answers.
"""
from __future__ import annotations

import json
import logging

from openai import APIError, APITimeoutError, AzureOpenAI

from .config import get_settings
from .models import Confidence, DocumentType

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You classify a single Indonesian document into exactly one \
type, using the OCR-extracted text provided. The text may be noisy (OCR \
artefacts, merged words, missing spaces); rely on the strongest signals.

Choose ONE document_type from:

- "ktp" — Kartu Tanda Penduduk (national identity card for ONE person). \
Signals: the header "PROVINSI ..." / "KARTU TANDA PENDUDUK", a single 16-digit \
"NIK", fields like "Tempat/Tgl Lahir", "Jenis Kelamin", "Alamat", "Agama", \
"Status Perkawinan", "Pekerjaan", "Kewarganegaraan", "Golongan Darah", \
"Berlaku Hingga". It describes ONE individual.

- "kk" — Kartu Keluarga (family card). Signals: the title "KARTU KELUARGA", a \
"No." family-card number, "Nama Kepala Keluarga", "Alamat", "Desa/Kelurahan", \
"Kecamatan", and a TABLE of family members with columns such as "Nama Lengkap", \
"NIK", "Jenis Kelamin", "Hubungan Dalam Keluarga", "Status Perkawinan", \
"Nama Orang Tua". It lists MULTIPLE people in a household.

- "sk" — Surat Keputusan / Surat Keterangan (an official decree or letter, \
e.g. an SK Pengangkatan / appointment letter). Signals: "SURAT KEPUTUSAN" or \
"SURAT KETERANGAN", a letter number ("Nomor: ..."), "Menimbang", "Mengingat", \
"Memutuskan", "Menetapkan", a letterhead and a signatory. It is prose/letter \
form, not an ID card and not a financial statement.

- "slip" — Slip Gaji (a salary / payroll slip). Signals: "SLIP GAJI" / "SLIP \
PEMBAYARAN GAJI", "Gaji Pokok", "Tunjangan", "Potongan", "Penghasilan", \
"Pendapatan", "Take Home Pay", "Gaji Bersih", a pay period, employee/NIK and \
earnings-minus-deductions layout.

- "mutasi" — Mutasi Rekening / Rekening Koran (a bank account statement). \
Signals: "MUTASI REKENING" / "REKENING KORAN", a bank name, account number, \
period, and a table of dated transactions with "Tanggal", "Keterangan", \
"Debit"/"Kredit"/"Mutasi", and a running "Saldo".

- "unknown" — none of the above, or the text is blank / illegible / \
insufficient to decide.

Set confidence to "high", "medium", or "low" based on how clearly the signals \
match. Use "low" and prefer "unknown" when the text is sparse or ambiguous. \
Give a one-line reasoning citing the decisive signal(s)."""


_RESPONSE_SCHEMA = {
    "name": "document_classification",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "document_type": {
                "type": "string",
                "enum": ["ktp", "kk", "sk", "slip", "mutasi", "unknown"],
            },
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "reasoning": {"type": "string"},
        },
        "required": ["document_type", "confidence", "reasoning"],
    },
    "strict": True,
}


def classify(text: str) -> tuple[DocumentType, Confidence, str, str | None]:
    """Classify OCR text. Returns ``(document_type, confidence, reasoning, error)``.

    ``error`` is ``None`` on success, or a short message when the LLM call /
    parse failed (in which case document_type is ``unknown``).
    """
    if not text.strip():
        return DocumentType.unknown, Confidence.low, "", "no text to classify"

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
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_schema", "json_schema": _RESPONSE_SCHEMA},
            temperature=0,
        )
        raw = completion.choices[0].message.content or "{}"
        decoded = json.loads(raw)
        doc_type = DocumentType(decoded["document_type"])
        confidence = Confidence(decoded["confidence"])
        reasoning = decoded.get("reasoning", "")
    except (APIError, APITimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("LLM classification failed: %s", exc)
        return DocumentType.unknown, Confidence.low, "", f"classifier error: {exc}"

    return doc_type, confidence, reasoning, None
