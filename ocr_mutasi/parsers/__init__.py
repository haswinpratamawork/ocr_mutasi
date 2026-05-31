"""Per-bank parsers.

`detect_bank` looks at the first page of OCR'd text and returns one of the
supported bank keys. `get_parser` returns the module that knows how to read
that bank's table layout. Each parser module exposes the same two functions
(`parse_header`, `parse_transactions`) so the pipeline stays generic.
"""
from __future__ import annotations

from types import ModuleType

from ..models import TextChunk
from . import bca, bri, mandiri
from .common import ParseResult

__all__ = ["ParseResult", "detect_bank", "get_parser"]


def detect_bank(chunks: list[TextChunk]) -> str:
    """Look at page-1 chunks for a known title/product signature.

    Returns one of {"BCA", "BRI", "Mandiri", "UNKNOWN"}.
    """
    page1_text = " ".join(c.text for c in chunks if c.page == 1).upper()
    if "REKENING TAHAPAN" in page1_text:
        return "BCA"
    if "LAPORAN TRANSAKSI FINANSIAL" in page1_text or "BRITAMA" in page1_text:
        return "BRI"
    # Mandiri e-Statement always carries the product label "Tabungan Mandiri"
    # and the bank's HQ address ("Menara Mandiri") on page 1 — either is a
    # clean disambiguator against the other two banks.
    if "TABUNGAN MANDIRI" in page1_text or "MENARA MANDIRI" in page1_text:
        return "Mandiri"
    return "UNKNOWN"


def get_parser(bank: str) -> ModuleType:
    if bank == "BCA":
        return bca
    if bank == "BRI":
        return bri
    if bank == "Mandiri":
        return mandiri
    raise ValueError(f"Unsupported bank: {bank!r}")
