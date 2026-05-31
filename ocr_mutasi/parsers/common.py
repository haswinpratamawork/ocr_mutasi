"""Shared types and helpers used by every bank parser."""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Optional

from ..models import TextChunk, Transaction


@dataclass
class ParseResult:
    transactions: list[Transaction]
    parse_warnings: list[str]
    balance_warnings: list[str]


@dataclass
class Row:
    """One visual line of text, bucketed into named columns.

    `cells` keys are parser-defined column names (e.g. ``"tanggal"`` /
    ``"keterangan"`` for BCA; ``"tanggal"`` / ``"uraian"`` / ``"teller"`` /
    ``"debet"`` / ``"kredit"`` / ``"saldo"`` for BRI).
    """
    cells: dict[str, str]
    page: int
    y: float

    def cell(self, name: str) -> str:
        return self.cells.get(name, "")


def extract_amount(text: str) -> Optional[float]:
    """Pull the first parseable number out of a cell.

    Handles Indonesian formatting (``,`` thousands, ``.`` decimals) and ignores
    trailing markers like ``DB``/``CR``. Returns None if no number is present.
    """
    if not text:
        return None
    for token in text.replace(",", "").split():
        try:
            return float(token)
        except ValueError:
            continue
    return None


def cluster_lines(chunks: list[TextChunk], tol_ratio: float = 0.6) -> list[list[TextChunk]]:
    """Group chunks into visual lines by Y-center.

    Tolerance is `tol_ratio * median(chunk_height)`, clipped to a small minimum
    so single-pixel baseline jitter doesn't fragment a line.
    """
    if not chunks:
        return []
    median_h = statistics.median(c.height for c in chunks) or 8.0
    tol = max(2.5, tol_ratio * median_h)
    lines: list[list[TextChunk]] = []
    for c in sorted(chunks, key=lambda c: c.y_center):
        if lines and abs(c.y_center - statistics.mean(b.y_center for b in lines[-1])) <= tol:
            lines[-1].append(c)
        else:
            lines.append([c])
    return lines


def join_cell(chunks: list[TextChunk]) -> str:
    """Left-to-right concatenation of a column's chunks on one line."""
    return " ".join(c.text for c in sorted(chunks, key=lambda c: c.x0)).strip()


def field_value(page_chunks: list[TextChunk], label: str) -> Optional[str]:
    """Find the text chunk immediately to the right of an exact label chunk on the same line."""
    labels = [c for c in page_chunks if c.text.strip().upper() == label.upper()]
    if not labels:
        return None
    anchor = labels[0]
    same_row = [c for c in page_chunks
                if c.x0 > anchor.x1 and abs(c.y_center - anchor.y_center) < 4.0
                and c.text.strip() not in {":", ""}]
    same_row.sort(key=lambda c: c.x0)
    return same_row[0].text.strip() if same_row else None


def year_from_periode(periode: Optional[str], default_today: bool = True) -> int:
    """First 4-digit token in PERIODE → int. Falls back to current year if absent."""
    from datetime import date
    if periode:
        for tok in periode.upper().replace("-", " ").replace("/", " ").split():
            if tok.isdigit() and len(tok) == 4:
                return int(tok)
            if tok.isdigit() and len(tok) == 2:  # BRI '26' → 2026
                return 2000 + int(tok)
    return date.today().year if default_today else 0
