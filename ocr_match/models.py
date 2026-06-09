"""Typed data structures for ocr_match.

These types double as FastAPI response schemas. Where they mirror types from
the two upstream services (`ocr_slip` and `ocr_mutasi`) we use ``extra="allow"``
so additional fields the upstream services add later flow through without
schema changes here.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------- upstream mirrors -----------------------------

class ParsedSlip(BaseModel):
    """Mirrors ``ocr_slip.ParsedDocument`` with a derived ``month`` field."""
    model_config = ConfigDict(extra="allow")

    source_file: str
    worker_name: Optional[str] = None
    institution_name: Optional[str] = None
    total_paid: Optional[float] = None
    pokok: float = 0.0
    tax: float = 0.0
    incentive: float = 0.0
    deduction: float = 0.0
    other_deduction: float = 0.0
    period: Optional[str] = Field(
        None,
        description="YYYY-MM emitted by ocr_slip from the slip's 'Period:' line. "
                    "When present, takes precedence over filename-month parsing.",
    )
    confidence_notes: list[str] = Field(default_factory=list)
    extraction_method: str = ""
    month: Optional[str] = Field(None, description="YYYY-MM, derived in pipeline")


class GajiCredit(BaseModel):
    """Mirrors ``ocr_mutasi.models.BatchClassifiedCredit`` for ``category=='Gaji'``."""
    model_config = ConfigDict(extra="allow")

    source_file: str
    tanggal: str
    keterangan: str
    amount: float
    type: str = "CR"
    saldo: Optional[float] = None
    cbg: Optional[str] = None
    page: int
    category: Optional[str] = None
    confidence: Optional[float] = None
    reason: Optional[str] = None
    month: Optional[str] = Field(None, description="YYYY-MM, derived in pipeline")


# ---------------------------- match result types ---------------------------

class MatchPair(BaseModel):
    """One slip paired with one bank-credit row."""
    slip: ParsedSlip
    credit: GajiCredit
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str
    amount_diff_rp: float = Field(..., description="credit.amount - slip.total_paid; signed")
    amount_diff_pct: float = Field(..., description="amount_diff_rp / slip.total_paid; signed")
    days_off: int = Field(0, description="day-of-month of the credit's tanggal")
    match_pattern: str = Field(
        ...,
        description="Which preference tier produced the pairing. "
                    "'next_month' = slip month X paid in bank month X+1 (Indonesian default); "
                    "'same_month' = paid in slip's own month X; "
                    "'future_month' = paid 2–3 months after the slip (delayed payroll); "
                    "'amount_only' = matched on amount alone, slip carried no period info.",
    )


class MatchAudit(BaseModel):
    slip_count: int
    credit_count: int
    matched_count: int
    months_processed: list[str] = Field(default_factory=list)
    matcher_errors: list[str] = Field(default_factory=list)
    upstream_errors: list[str] = Field(default_factory=list)


class MatchResponse(BaseModel):
    matches: list[MatchPair]
    unmatched_slips: list[ParsedSlip]
    unmatched_credits: list[GajiCredit]
    audit: MatchAudit
