"""Parser for Mandiri 'Tabungan Mandiri' e-Statement PDFs.

5 columns:  No  |  Tanggal/Date  |  Keterangan/Remarks  |  Nominal (IDR)/Amount  |  Saldo (IDR)/Balance

Differences from BCA and BRI that the implementation has to handle:

* **Indonesian number format** — Mandiri uses ``.`` as the thousands separator
  and ``,`` as the decimal separator (e.g. ``5.000.000,00`` = 5,000,000.00).
  The other two parsers' ``extract_amount`` assumes the opposite, so we use
  our own ``_parse_id_amount`` here.
* **Sign-based debit/credit** — the Nominal column carries a ``+`` or ``-``
  prefix (e.g. ``+5.000.000,00`` / ``-2.500,00``). No separate columns and no
  ``DB``/``CR`` suffix.
* **Multi-line transactions** — each transaction spans several visual lines.
  A typical row looks like::

      Transfer BI Fast                                          ← Keterangan main label
      01 Apr 2026                                               ← Tanggal date
      1   Dari BRI            +5.000.000,00      6.000.000,00   ← anchor line (No + values)
          08:56:51 WIB                                          ← Tanggal time
          BUDI SANTOSO 1234567890123                  ← Keterangan continuation

  Lines whose y-coordinates are within ``TX_GAP_THRESHOLD_PT`` belong to the
  same transaction; a larger gap starts a new block.
* **Dates** are ``DD MMM YYYY`` with English month abbreviations
  (``Apr``, ``May``, ``Jun`` …). Indonesian abbreviations are accepted too.
* **Account name** in the header may be split across two chunks
  (``BUDI`` on one line, ``SANTOSO`` on the next).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..models import AccountHeader, TextChunk, Transaction, TxType
from .common import ParseResult, Row, cluster_lines, join_cell

# Column header tokens — Mandiri's Indonesian line above the English subheader.
COLUMN_NAMES: tuple[str, ...] = ("No", "Tanggal", "Keterangan", "Nominal (IDR)", "Saldo (IDR)")

# Date row pattern: DD MMM YYYY (English or Indonesian month abbreviation).
DATE_RE = re.compile(r"^\s*(\d{1,2})\s+([A-Za-z]{3,})\s+(\d{4})\s*$")

# Month abbreviations — English (Mandiri's default) plus common Indonesian variants.
MONTHS: dict[str, int] = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "MEI": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "AGU": 8, "AGT": 8, "SEP": 9, "OCT": 10, "OKT": 10,
    "NOV": 11, "DEC": 12, "DES": 12,
}

# Gap (in PDF points) above which two consecutive visual lines belong to
# different transactions. Calibrated from the included sample: intra-tx gaps
# are 4–6 pt, inter-tx gaps are 30+ pt.
TX_GAP_THRESHOLD_PT = 18.0

# Markers that indicate we've hit the boilerplate at the bottom of the
# statement and should stop processing.
SUMMARY_MARKERS: tuple[str, ...] = (
    "disclaimer",
    "ini adalah batas akhir transaksi anda",
)


@dataclass
class _MandiriLayout:
    no: tuple[float, float]
    tanggal: tuple[float, float]
    keterangan: tuple[float, float]
    nominal: tuple[float, float]
    saldo: tuple[float, float]


# --------------------------- public entry points --------------------------

def parse_header(chunks: list[TextChunk]) -> AccountHeader:
    page1 = [c for c in chunks if c.page == 1]
    return AccountHeader(
        bank="Mandiri",
        no_rekening=_mandiri_field(page1, "Nomor Rekening/"),
        nama=_mandiri_field(page1, "Nama/"),
        periode=_mandiri_field(page1, "Periode/"),
        mata_uang=_mandiri_field(page1, "Mata Uang/"),
    )


def parse_transactions(chunks: list[TextChunk], header: AccountHeader) -> ParseResult:
    transactions: list[Transaction] = []
    parse_warnings: list[str] = []
    balance_warnings: list[str] = []
    last_layout: Optional[_MandiriLayout] = None
    running_balance: Optional[float] = None

    by_page: dict[int, list[TextChunk]] = {}
    for c in chunks:
        by_page.setdefault(c.page, []).append(c)

    for page in sorted(by_page):
        page_chunks = by_page[page]
        layout, header_y = _detect_column_layout(page_chunks)
        if layout is None:
            if last_layout is None:
                continue
            layout, header_y = last_layout, -1.0
        else:
            last_layout = layout

        body = [c for c in page_chunks if c.y0 > header_y + 1]
        body = _trim_at_disclaimer(body)
        rows = _chunks_to_rows(body, layout, page)
        for block in _group_into_blocks(rows):
            tx, warn = _block_to_transaction(block)
            if warn:
                parse_warnings.append(f"p{page} y={block[0].y:.0f}: {warn}")
            if tx is None:
                continue
            transactions.append(tx)
            delta = tx.amount if tx.type == "CR" else -tx.amount
            if running_balance is None:
                if tx.saldo is not None:
                    running_balance = tx.saldo
            else:
                running_balance += delta
                if tx.saldo is not None and abs(running_balance - tx.saldo) > 0.5:
                    balance_warnings.append(
                        f"p{page} {tx.tanggal}: running {running_balance:.2f} vs printed {tx.saldo:.2f}"
                    )
                    running_balance = tx.saldo

    return ParseResult(transactions, parse_warnings, balance_warnings)


# --------------------------- header / column layout -----------------------

def _detect_column_layout(page_chunks: list[TextChunk]) -> tuple[Optional[_MandiriLayout], float]:
    """Locate the Indonesian column-header row and snapshot per-column x-spans."""
    candidates = {name: [c for c in page_chunks if c.text.strip() == name]
                  for name in COLUMN_NAMES}
    if not all(candidates.values()):
        return None, 0.0
    for no in candidates["No"]:
        picked: dict[str, TextChunk] = {"No": no}
        ok = True
        for name in COLUMN_NAMES[1:]:
            match = next((c for c in candidates[name] if abs(c.y_center - no.y_center) < 3.0), None)
            if match is None:
                ok = False
                break
            picked[name] = match
        if not ok:
            continue
        layout = _MandiriLayout(
            no=(picked["No"].x0, picked["No"].x1),
            tanggal=(picked["Tanggal"].x0, picked["Tanggal"].x1),
            keterangan=(picked["Keterangan"].x0, picked["Keterangan"].x1),
            nominal=(picked["Nominal (IDR)"].x0, picked["Nominal (IDR)"].x1),
            saldo=(picked["Saldo (IDR)"].x0, picked["Saldo (IDR)"].x1),
        )
        # Include the English subheader (~10 pt below) in the header band so
        # the first transaction row isn't clustered with it.
        return layout, max(picked[n].y1 for n in COLUMN_NAMES) + 12.0
    return None, 0.0


def _column_boundaries(layout: _MandiriLayout) -> tuple[float, float, float, float]:
    """X-thresholds calibrated for Mandiri Tabungan.

    * No / Tanggal     — just left of Tanggal's header.
    * Tanggal / Ket    — just left of Keterangan's header.
    * Ket / Nominal    — well left of Nominal's header because Keterangan
                         content can wrap quite far right (counter-party name
                         lines run out to xc≈290).
    * Nominal / Saldo  — midpoint between Nominal's right edge and Saldo's
                         left edge; both columns hold right-aligned amounts.
    """
    return (
        layout.tanggal[0] - 10.0,
        layout.keterangan[0] - 5.0,
        layout.nominal[0] - 30.0,
        (layout.nominal[1] + layout.saldo[0]) / 2,
    )


def _column_for_x(x: float, layout: _MandiriLayout) -> str:
    b1, b2, b3, b4 = _column_boundaries(layout)
    if x < b1: return "no"
    if x < b2: return "tanggal"
    if x < b3: return "keterangan"
    if x < b4: return "nominal"
    return "saldo"


# --------------------------- row / block assembly -------------------------

def _trim_at_disclaimer(chunks: list[TextChunk]) -> list[TextChunk]:
    """Drop any chunks at/below the disclaimer / end-of-transaction marker."""
    marker_ys = [c.y0 for c in chunks if c.text.strip().lower() in SUMMARY_MARKERS]
    if not marker_ys:
        return chunks
    cutoff = min(marker_ys)
    return [c for c in chunks if c.y0 < cutoff - 1]


def _chunks_to_rows(chunks: list[TextChunk], layout: _MandiriLayout, page: int) -> list[Row]:
    rows: list[Row] = []
    for line in cluster_lines(chunks):
        per_col: dict[str, list[TextChunk]] = {n: [] for n in ("no", "tanggal", "keterangan", "nominal", "saldo")}
        for c in line:
            per_col[_column_for_x(c.x_center, layout)].append(c)
        rows.append(Row(
            cells={n: join_cell(per_col[n]) for n in per_col},
            page=page,
            y=sum(c.y_center for c in line) / len(line),
        ))
    return rows


def _group_into_blocks(rows: list[Row]) -> list[list[Row]]:
    """Each transaction spans 3–5 visual lines packed close together. A gap
    larger than ``TX_GAP_THRESHOLD_PT`` between consecutive lines marks the
    boundary between two transactions."""
    if not rows:
        return []
    rows = sorted(rows, key=lambda r: r.y)
    blocks: list[list[Row]] = [[rows[0]]]
    for prev, curr in zip(rows, rows[1:]):
        if (curr.y - prev.y) > TX_GAP_THRESHOLD_PT:
            blocks.append([curr])
        else:
            blocks[-1].append(curr)
    return blocks


def _block_to_transaction(block: list[Row]) -> tuple[Optional[Transaction], Optional[str]]:
    """A real transaction has: a sequential ``No``, a ``DD MMM YYYY`` date,
    and a signed amount in Nominal. Anything missing one of those is treated
    as boilerplate and silently dropped."""
    no_parts = [r.cell("no") for r in block if r.cell("no")]
    no_text = " ".join(no_parts).strip()
    if not no_text or not no_text.isdigit():
        return None, None  # not a transaction row

    # Date — scan rows for the first one whose Tanggal cell matches DD MMM YYYY.
    date_parts: Optional[tuple[int, int, int]] = None
    for r in block:
        date_parts = _parse_date(r.cell("tanggal"))
        if date_parts:
            break
    if date_parts is None:
        return None, "no parseable date in block"
    day, month, year = date_parts

    # Amount — first row with a parseable Nominal value, including its sign.
    amount: Optional[float] = None
    sign: Optional[str] = None
    for r in block:
        amount, sign = _parse_signed_amount(r.cell("nominal"))
        if amount is not None:
            break
    if amount is None:
        return None, "no parseable amount in block"
    tx_type: TxType = "DB" if sign == "-" else "CR"

    # Saldo — first row with a Saldo value (unsigned).
    saldo: Optional[float] = None
    for r in block:
        s = _parse_id_amount(r.cell("saldo"))
        if s is not None:
            saldo = s
            break

    # Description = every Keterangan cell across the block, joined left-to-right.
    desc_parts = [r.cell("keterangan") for r in block if r.cell("keterangan")]
    keterangan = " | ".join(desc_parts)

    return Transaction(
        tanggal=f"{year:04d}-{month:02d}-{day:02d}",
        keterangan=keterangan,
        cbg=None,
        amount=amount,
        type=tx_type,
        saldo=saldo,
        page=block[0].page,
    ), None


# --------------------------- helpers --------------------------------------

def _parse_date(text: str) -> Optional[tuple[int, int, int]]:
    m = DATE_RE.match(text or "")
    if not m:
        return None
    day = int(m.group(1))
    mon_name = m.group(2).upper()[:3]
    year = int(m.group(3))
    if mon_name not in MONTHS or not (1 <= day <= 31):
        return None
    return day, MONTHS[mon_name], year


def _parse_id_amount(text: str) -> Optional[float]:
    """Parse an Indonesian-format number: ``.`` thousands, ``,`` decimals.
    Strips a leading ``+``/``-`` and returns the absolute value."""
    if not text:
        return None
    for token in text.split():
        t = token.lstrip("+-")
        cleaned = t.replace(".", "").replace(",", ".")
        try:
            return float(cleaned)
        except ValueError:
            continue
    return None


def _parse_signed_amount(text: str) -> tuple[Optional[float], Optional[str]]:
    """Same as ``_parse_id_amount`` but also returns the sign (``+``/``-``/None)."""
    if not text:
        return None, None
    for token in text.split():
        sign: Optional[str] = None
        t = token
        if t.startswith("+"):
            sign, t = "+", t[1:]
        elif t.startswith("-"):
            sign, t = "-", t[1:]
        cleaned = t.replace(".", "").replace(",", ".")
        try:
            return float(cleaned), sign
        except ValueError:
            continue
    return None, None


def _mandiri_field(page_chunks: list[TextChunk], label: str) -> Optional[str]:
    """Extract the value(s) belonging to a Mandiri header label like ``Nama/``.

    Mandiri's header is a two-column layout (account info on the left, period
    info on the right). For each label we look in the matching value column
    within a small y-window — that picks up the first value plus any
    continuation line (e.g. ``SANTOSO`` underneath ``BUDI``).
    """
    labels = [c for c in page_chunks if c.text.strip() == label]
    if not labels:
        return None
    anchor = labels[0]
    # Left-column labels sit at x ≈ 16; right-column ones at x ≈ 233.
    x_range = (100.0, 230.0) if anchor.x0 < 200 else (330.0, 440.0)
    candidates = [c for c in page_chunks
                  if x_range[0] <= c.x0 < x_range[1]
                  and anchor.y0 - 2 <= c.y0 <= anchor.y0 + 18
                  and c.text.strip() not in {":", ""}]
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c.y0, c.x0))
    return " ".join(c.text.strip() for c in candidates).strip() or None
