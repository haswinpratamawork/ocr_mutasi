"""Per-bank parsers.

`detect_bank` looks at the first page of OCR'd text and returns one of the
supported bank keys. `get_parser` returns the module that knows how to read
that bank's table layout. Each parser module exposes the same two functions
(`parse_header`, `parse_transactions`) so the pipeline stays generic.
"""
from __future__ import annotations

from types import ModuleType

from ..models import TextChunk
from . import bca, bri, mandiri, permata, sinarmas
from .common import ParseResult

__all__ = ["ParseResult", "SUPPORTED_BANKS", "detect_bank", "get_parser"]

SUPPORTED_BANKS: tuple[str, ...] = (
    "BCA Rekening Tahapan",
    "BRI BritAma",
    "Mandiri Tabungan",
    "Permata Rekening Koran",
    "Sinarmas Tabungan",
)


def detect_bank(chunks: list[TextChunk]) -> str:
    """Look at page-1 chunks for a known title/product signature.

    Returns one of {"BCA", "BRI", "Mandiri", "Permata", "Sinarmas", "UNKNOWN"}.
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
    # Permata's footer carries the company name; the URL/phone block is the
    # cleanest disambiguator (the title 'Rekening Koran' is generic and could
    # collide with other banks' future statements).
    if ("PERMATABANK.COM" in page1_text
            or "PT BANK PERMATA" in page1_text
            or "PERMATABANK" in page1_text):
        return "Permata"
    # Sinarmas pages carry the product label "Tabungan Sinarmas <...>" in the
    # account-info footer (e.g. ``Tabungan Sinarmas Payroll Premium``). The
    # bank's name as a bare token also appears on every page. ``SINARMAS``
    # alone disambiguates since none of the other supported banks use it.
    if "SINARMAS" in page1_text or "BANK SINARMAS" in page1_text:
        return "Sinarmas"
    return "UNKNOWN"


def get_parser(bank: str) -> ModuleType:
    if bank == "BCA":
        return bca
    if bank == "BRI":
        return bri
    if bank == "Mandiri":
        return mandiri
    if bank == "Permata":
        return permata
    if bank == "Sinarmas":
        return sinarmas
    raise ValueError(f"Unsupported bank: {bank!r}")
