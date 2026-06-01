"""Typed data structures shared across the pipeline and API.

All models are Pydantic so the same definitions serve as internal types and as
FastAPI response schemas.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

TxType = Literal["DB", "CR"]
Category = Literal["Gaji", "THR", "Bonus", "Insentif", "Lainnya"]


# ----------------------------- low-level chunks -----------------------------

class TextChunk(BaseModel):
    """One contiguous run of text from the PDF text layer, with its bounding box.

    Coordinates are in PDF points with the origin at the TOP-LEFT (y grows down),
    which differs from PDFium's native bottom-up convention. The conversion
    happens in `pdf_extractor`.
    """
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    page: int

    @property
    def x_center(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def y_center(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def height(self) -> float:
        return self.y1 - self.y0


# ----------------------------- domain transactions --------------------------
# (Internal column layouts and row types now live in each parser module so
# BCA's 5-column layout and BRI's 6-column layout can diverge cleanly.)


class Transaction(BaseModel):
    """One parsed bank-statement transaction."""
    tanggal: str = Field(..., description="ISO date, e.g. 2026-04-15")
    keterangan: str
    cbg: Optional[str] = None
    amount: float
    type: TxType
    saldo: Optional[float] = None
    page: int


class ClassifiedCredit(Transaction):
    """A credit transaction with LLM-assigned category."""
    category: Optional[Category] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    reason: Optional[str] = None


# ----------------------------- account header -------------------------------

class AccountHeader(BaseModel):
    bank: str = "BCA"
    no_rekening: Optional[str] = None
    nama: Optional[str] = None
    periode: Optional[str] = Field(None, description="e.g. APRIL 2026")
    mata_uang: Optional[str] = None


# ----------------------------- response shape -------------------------------

class Audit(BaseModel):
    pages_processed: int
    rows_detected: int
    credit_count: int
    debit_count: int
    balance_warnings: list[str] = []
    parse_warnings: list[str] = []
    classifier_errors: list[str] = []


class ExtractionResponse(BaseModel):
    account: AccountHeader
    transactions: list[Transaction]
    credits: list[ClassifiedCredit]
    audit: Audit


# ----------------------------- batch response shape ------------------------

class FileExtraction(BaseModel):
    """Per-file extraction inside a batch response."""
    filename: str
    account: AccountHeader
    transactions: list[Transaction]
    audit: Audit


class BatchClassifiedCredit(ClassifiedCredit):
    """A classified credit annotated with the file it came from (for batch responses)."""
    source_file: str


class CategoryTotal(BaseModel):
    """Per-category roll-up for a batch response.

    `min` is the smallest single-transaction amount in the category (useful
    for sanity-checking salary floors, smallest bonus, etc.); ``null`` when
    the category has no transactions.
    """
    count: int
    sum: float
    min: Optional[float] = None


class BatchAudit(BaseModel):
    files_processed: int
    transactions_total: int
    credits_total: int
    classifier_errors: list[str] = []
    category_totals: dict[str, CategoryTotal] = {}


class BatchExtractionResponse(BaseModel):
    """Response from /api/v1/mutations/extract-batch.

    Per-file data lives in `files`. All credits across all files are
    classified together (cross-month context) and returned flat in `credits`,
    with `source_file` indicating which PDF each came from.
    """
    files: list[FileExtraction]
    credits: list[BatchClassifiedCredit]
    audit: BatchAudit
