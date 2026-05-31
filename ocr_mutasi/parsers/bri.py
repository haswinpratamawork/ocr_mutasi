"""Parser for BRI 'BritAma' / 'LAPORAN TRANSAKSI FINANSIAL' statements.

6 columns (Indonesian header above an English subheader):
    Tanggal Transaksi  |  Uraian Transaksi  |  Teller  |  Debet  |  Kredit  |  Saldo
    Transaction Date   |  Transaction Desc  |  User ID |  Debit  |  Credit |  Balance

Each transaction occupies one row whose date column matches
``DD/MM/YY HH:MM:SS``; description-continuation lines (e.g. counter-party
name on the next baseline) belong to the preceding transaction. Debet and
Kredit are SEPARATE columns; the "other" column carries a ``0.00`` placeholder.

A summary block at the bottom of the final page lists Saldo Awal / Total
Debet / Total Kredit / Saldo Akhir; we explicitly stop processing rows when
we encounter that block so its numbers don't pollute the last transaction.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..models import AccountHeader, TextChunk, Transaction, TxType
from .common import (
    ParseResult,
    Row,
    cluster_lines,
    extract_amount,
    field_value,
    join_cell,
    year_from_periode,
)

# The Indonesian header line — we anchor on this and ignore the English line below.
COLUMN_NAMES_ID: tuple[str, ...] = (
    "Tanggal Transaksi", "Uraian Transaksi", "Teller", "Debet", "Kredit", "Saldo"
)

# DD/MM/YY HH:MM:SS — BRI's row date.
DATE_TIME_RE = re.compile(r"^\s*(\d{1,2})/(\d{1,2})/(\d{2,4})\s+(\d{1,2}):(\d{2}):(\d{2})\s*$")

# Tokens that mark the summary block at the bottom of the final page.
SUMMARY_MARKERS: tuple[str, ...] = (
    "saldo awal", "opening balance",
    "total transaksi debet", "total debit transaction",
    "total transaksi kredit", "total credit transaction",
    "saldo akhir", "closing balance",
)


@dataclass
class _BriLayout:
    tanggal: tuple[float, float]
    uraian: tuple[float, float]
    teller: tuple[float, float]
    debet: tuple[float, float]
    kredit: tuple[float, float]
    saldo: tuple[float, float]


# --------------------------- public entry points --------------------------

def parse_header(chunks: list[TextChunk]) -> AccountHeader:
    page1 = [c for c in chunks if c.page == 1]
    return AccountHeader(
        bank="BRI",
        no_rekening=field_value(page1, "No. Rekening") or field_value(page1, "Account No"),
        nama=_holder_name(page1),
        periode=field_value(page1, "Periode Transaksi") or field_value(page1, "Transaction Period"),
        mata_uang=field_value(page1, "Valuta") or field_value(page1, "Currency"),
    )


def parse_transactions(chunks: list[TextChunk], header: AccountHeader) -> ParseResult:
    year = year_from_periode(header.periode)
    transactions: list[Transaction] = []
    parse_warnings: list[str] = []
    balance_warnings: list[str] = []
    last_layout: Optional[_BriLayout] = None
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
        body = _trim_at_summary(body)
        rows = _chunks_to_rows(body, layout, page)
        for block in _group_into_blocks(rows):
            tx, warn = _block_to_transaction(block, year)
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

def _detect_column_layout(page_chunks: list[TextChunk]) -> tuple[Optional[_BriLayout], float]:
    """Find the Indonesian header row containing all 6 column labels."""
    candidates = {name: [c for c in page_chunks if c.text.strip() == name]
                  for name in COLUMN_NAMES_ID}
    if not all(candidates.values()):
        return None, 0.0
    for tanggal in candidates["Tanggal Transaksi"]:
        picked: dict[str, TextChunk] = {"Tanggal Transaksi": tanggal}
        ok = True
        for name in COLUMN_NAMES_ID[1:]:
            match = next((c for c in candidates[name] if abs(c.y_center - tanggal.y_center) < 3.0), None)
            if match is None:
                ok = False
                break
            picked[name] = match
        if not ok:
            continue
        layout = _BriLayout(
            tanggal=(picked["Tanggal Transaksi"].x0, picked["Tanggal Transaksi"].x1),
            uraian=(picked["Uraian Transaksi"].x0, picked["Uraian Transaksi"].x1),
            teller=(picked["Teller"].x0, picked["Teller"].x1),
            debet=(picked["Debet"].x0, picked["Debet"].x1),
            kredit=(picked["Kredit"].x0, picked["Kredit"].x1),
            saldo=(picked["Saldo"].x0, picked["Saldo"].x1),
        )
        # Header bottom — include the English subheader (~10pt below)
        # so transaction rows aren't accidentally clustered with it.
        return layout, max(picked[n].y1 for n in COLUMN_NAMES_ID) + 12.0
    return None, 0.0


def _column_boundaries(layout: _BriLayout) -> tuple[float, float, float, float, float]:
    """X-thresholds calibrated for BRI BritAma.

    * TANGGAL/URAIAN: just past the date column header (description content
      starts slightly left of its centered URAIAN header).
    * URAIAN/TELLER : just left of the (narrow) TELLER header.
    * TELLER/DEBET  : DEBET amount content starts ~12pt left of its header.
    * DEBET/KREDIT  : midpoint between adjacent header centers — works because
      both columns hold right-aligned amounts roughly equidistant from their
      header centers.
    * KREDIT/SALDO  : same midpoint logic.
    """
    debet_c = (layout.debet[0] + layout.debet[1]) / 2
    kredit_c = (layout.kredit[0] + layout.kredit[1]) / 2
    saldo_c = (layout.saldo[0] + layout.saldo[1]) / 2
    return (
        layout.tanggal[1] + 5.0,
        layout.teller[0] - 10.0,
        layout.debet[0] - 12.0,
        (debet_c + kredit_c) / 2,
        (kredit_c + saldo_c) / 2,
    )


def _column_for_x(x: float, layout: _BriLayout) -> str:
    b1, b2, b3, b4, b5 = _column_boundaries(layout)
    if x < b1: return "tanggal"
    if x < b2: return "uraian"
    if x < b3: return "teller"
    if x < b4: return "debet"
    if x < b5: return "kredit"
    return "saldo"


# --------------------------- row / block assembly -------------------------

def _trim_at_summary(chunks: list[TextChunk]) -> list[TextChunk]:
    """Drop any chunks at/below the start of the final-page summary block."""
    summary_ys = [c.y0 for c in chunks if c.text.strip().lower() in SUMMARY_MARKERS]
    if not summary_ys:
        return chunks
    cutoff = min(summary_ys)
    return [c for c in chunks if c.y0 < cutoff - 1]


def _chunks_to_rows(chunks: list[TextChunk], layout: _BriLayout, page: int) -> list[Row]:
    rows: list[Row] = []
    for line in cluster_lines(chunks):
        per_col: dict[str, list[TextChunk]] = {n: [] for n in ("tanggal", "uraian", "teller", "debet", "kredit", "saldo")}
        for c in line:
            per_col[_column_for_x(c.x_center, layout)].append(c)
        rows.append(Row(
            cells={n: join_cell(per_col[n]) for n in per_col},
            page=page,
            y=sum(c.y_center for c in line) / len(line),
        ))
    return rows


def _looks_like_tx_header(row: Row) -> bool:
    """A real BRI transaction row has a DD/MM/YY HH:MM:SS date AND at least
    one amount (debet or kredit). The latter is critical to avoid promoting
    isolated header fragments to transactions."""
    if _parse_datetime(row.cell("tanggal")) is None:
        return False
    return extract_amount(row.cell("debet")) is not None or extract_amount(row.cell("kredit")) is not None


def _group_into_blocks(rows: list[Row]) -> list[list[Row]]:
    blocks: list[list[Row]] = []
    for row in rows:
        if _looks_like_tx_header(row):
            blocks.append([row])
        elif blocks:
            blocks[-1].append(row)
        # rows before the first transaction are dropped silently
    return blocks


def _block_to_transaction(block: list[Row], year: int) -> tuple[Optional[Transaction], Optional[str]]:
    head = block[0]
    if not _looks_like_tx_header(head):
        return None, None
    dmy = _parse_datetime(head.cell("tanggal"))
    assert dmy is not None
    day, month, parsed_year = dmy
    # If the row's own 2-digit year disagrees with the periode year, prefer the row's.
    effective_year = parsed_year if parsed_year >= 2000 else year

    debet = extract_amount(head.cell("debet")) or 0.0
    kredit = extract_amount(head.cell("kredit")) or 0.0
    # BRI uses 0.00 as a placeholder. The real amount is whichever is non-zero.
    if kredit > 0 and debet == 0:
        amount, tx_type = kredit, "CR"
    elif debet > 0 and kredit == 0:
        amount, tx_type = debet, "DB"
    elif debet > 0 and kredit > 0:
        # Both non-zero — extremely rare. Treat as warning and pick the larger.
        amount = max(debet, kredit)
        tx_type = "CR" if kredit >= debet else "DB"
    else:
        return None, "row has neither debet nor kredit"

    desc_parts = [head.cell("uraian")] + [r.cell("uraian") for r in block[1:] if r.cell("uraian")]
    keterangan = " | ".join(p for p in desc_parts if p)

    teller_parts = [head.cell("teller")] + [r.cell("teller") for r in block[1:] if r.cell("teller")]
    teller = " ".join(p for p in teller_parts if p) or None

    saldo = extract_amount(head.cell("saldo"))
    if saldo is None:
        for r in block[1:]:
            s = extract_amount(r.cell("saldo"))
            if s is not None:
                saldo = s
                break

    return Transaction(
        tanggal=f"{effective_year:04d}-{month:02d}-{day:02d}",
        keterangan=keterangan,
        cbg=teller,  # we surface BRI's teller/User-ID here; closest analogue to BCA's CBG
        amount=amount,
        type=tx_type,  # type: ignore[arg-type]
        saldo=saldo,
        page=head.page,
    ), None


# --------------------------- helpers --------------------------------------

def _parse_datetime(text: str) -> Optional[tuple[int, int, int]]:
    m = DATE_TIME_RE.match(text or "")
    if not m:
        return None
    d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return None
    if y < 100:  # 2-digit year → 2000+
        y += 2000
    return d, mo, y


def _holder_name(page_chunks: list[TextChunk]) -> Optional[str]:
    """BRI puts the holder name on the line ABOVE the address. The address
    line for personal accounts starts with ``JL``/``KOMP``/``GG``/``KOTA``/
    ``KOMPLEK``/``RT``, so we anchor on that.
    """
    left = [c for c in page_chunks if c.x0 < 280]
    address_anchors = [c for c in left
                       if c.text.strip().upper().startswith(
                           ("JL ", "JL.", "KOMP", "GG ", "KOTA ", "RT", "JALAN"))]
    if not address_anchors:
        return None
    first_y = min(c.y0 for c in address_anchors)
    # The name chunk usually sits 15–40 pt above the first address line.
    above = [c for c in left if c.y0 < first_y and (first_y - c.y0) < 60
             and c.text.strip() and c.text.strip().upper() == c.text.strip()
             and not c.text.strip().endswith(":")
             and not c.text.strip().endswith("/")]
    if not above:
        return None
    above.sort(key=lambda c: c.y0, reverse=True)
    # Skip generic labels like "Kepada Yth." / "To :"
    for c in above:
        t = c.text.strip()
        if any(t.upper().startswith(p) for p in ("KEPADA", "TO :", "TO:", "DEAR")):
            continue
        if len(t) >= 4:
            return t
    return above[0].text.strip()
