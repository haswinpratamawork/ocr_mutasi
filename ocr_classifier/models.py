"""Typed data structures shared across the pipeline and API.

All models are Pydantic so the same definitions serve as internal types and
as FastAPI response schemas.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """The five recognised Indonesian document types, plus an escape hatch."""

    ktp = "ktp"        # Kartu Tanda Penduduk (national ID)
    kk = "kk"          # Kartu Keluarga (family card)
    sk = "sk"          # Surat Keputusan / Surat Keterangan (decree/letter)
    slip = "slip"      # Slip Gaji (salary slip)
    mutasi = "mutasi"  # Mutasi Rekening / Rekening Koran (bank statement)
    unknown = "unknown"  # none of the above / blank / illegible


class Confidence(str, Enum):
    """Coarse, self-reported model confidence. Advisory, not calibrated."""

    high = "high"
    medium = "medium"
    low = "low"


class OcrAudit(BaseModel):
    """Trace of the OCR step plus any non-fatal problems for this document."""

    ocr_request_id: Optional[str] = None
    ocr_response_time_ms: Optional[float] = None
    extracted_char_count: int = 0
    page_count: int = 0
    errors: list[str] = Field(default_factory=list)


class ClassificationResult(BaseModel):
    """The classification of a single document."""

    filename: str
    document_type: DocumentType
    confidence: Confidence
    reasoning: str = ""
    audit: OcrAudit
    # OCR text echoed back only when the caller passes ?include_text=true.
    text: Optional[str] = None


class BatchClassificationResponse(BaseModel):
    """Response from /classify-batch — one result per uploaded file."""

    count: int
    results: list[ClassificationResult]
