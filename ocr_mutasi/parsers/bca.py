"""Parser for BCA 'Rekening Tahapan' statements.

5 columns: TANGGAL / KETERANGAN / CBG / MUTASI / SALDO. DB rows carry a trailing
``DB`` token in MUTASI; CR rows omit any suffix. Each transaction spans one
"header" row (date + amount) plus zero or more KETERANGAN detail rows below it.
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

COLUMN_NAMES: tuple[str, ...] = ("TANGGAL", "KETERANGAN", "CBG", "MUTASI", "SALDO")
DATE_RE = re.compile(r"^\s*(\d{1,2})/(\d{1,2})\s*$")


@dataclass
class _BcaLayout:
    tanggal: tuple[float, float]
    keterangan: tuple[float, float]
    cbg: tuple[float, float]
    mutasi: tuple[float, float]
    saldo: tuple[float, float]


# --------------------------- public entry points --------------------------

def parse_header(chunks: list[TextChunk]) -> AccountHeader:
    page1 = [c for c in chunks if c.page == 1]
    return AccountHeader(
        bank="BCA",
        no_rekening=field_value(page1, "NO. REKENING"),
        nama=_first_name_line(page1),
        periode=field_value(page1, "PERIODE"),
        mata_uang=field_value(page1, "MATA UANG"),
    )


def parse_transactions(chunks: list[TextChunk], header: AccountHeader) -> ParseResult:
    year = year_from_periode(header.periode)
    transactions: list[Transaction] = []
    parse_warnings: list[str] = []
    balance_warnings: list[str] = []
    last_layout: Optional[_BcaLayout] = None
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

def _detect_column_layout(page_chunks: list[TextChunk]) -> tuple[Optional[_BcaLayout], float]:
    candidates = {name: [c for c in page_chunks if c.text.strip().upper() == name]
                  for name in COLUMN_NAMES}
    if not all(candidates.values()):
        return None, 0.0
    for tanggal in candidates["TANGGAL"]:
        picked: dict[str, TextChunk] = {"TANGGAL": tanggal}
        ok = True
        for name in COLUMN_NAMES[1:]:
            match = next((c for c in candidates[name] if abs(c.y_center - tanggal.y_center) < 3.0), None)
            if match is None:
                ok = False
                break
            picked[name] = match
        if not ok:
            continue
        layout = _BcaLayout(
            tanggal=(picked["TANGGAL"].x0, picked["TANGGAL"].x1),
            keterangan=(picked["KETERANGAN"].x0, picked["KETERANGAN"].x1),
            cbg=(picked["CBG"].x0, picked["CBG"].x1),
            mutasi=(picked["MUTASI"].x0, picked["MUTASI"].x1),
            saldo=(picked["SALDO"].x0, picked["SALDO"].x1),
        )
        return layout, max(picked[n].y1 for n in COLUMN_NAMES)
    return None, 0.0


def _column_boundaries(layout: _BcaLayout) -> tuple[float, float, float, float]:
    """X-thresholds calibrated for BCA — see architecture.md §3.2 for rationale."""
    return (
        layout.tanggal[1] + 8.0,
        layout.cbg[0] - 3.0,
        layout.mutasi[0] - 8.0,
        layout.saldo[0] - 10.0,
    )


def _column_for_x(x: float, layout: _BcaLayout) -> str:
    b1, b2, b3, b4 = _column_boundaries(layout)
    if x < b1: return "tanggal"
    if x < b2: return "keterangan"
    if x < b3: return "cbg"
    if x < b4: return "mutasi"
    return "saldo"


# --------------------------- row / block assembly -------------------------

def _chunks_to_rows(chunks: list[TextChunk], layout: _BcaLayout, page: int) -> list[Row]:
    rows: list[Row] = []
    for line in cluster_lines(chunks):
        per_col: dict[str, list[TextChunk]] = {n: [] for n in ("tanggal", "keterangan", "cbg", "mutasi", "saldo")}
        for c in line:
            per_col[_column_for_x(c.x_center, layout)].append(c)
        rows.append(Row(
            cells={n: join_cell(per_col[n]) for n in per_col},
            page=page,
            y=sum(c.y_center for c in line) / len(line),
        ))
    return rows


def _looks_like_tx_header(row: Row) -> bool:
    return _parse_date(row.cell("tanggal")) is not None and extract_amount(row.cell("mutasi")) is not None


def _group_into_blocks(rows: list[Row]) -> list[list[Row]]:
    blocks: list[list[Row]] = []
    for row in rows:
        if _looks_like_tx_header(row):
            blocks.append([row])
        elif blocks:
            blocks[-1].append(row)
        else:
            blocks.append([row])  # pre-table rows (SALDO AWAL) — won't parse as tx
    return blocks


def _block_to_transaction(block: list[Row], year: int) -> tuple[Optional[Transaction], Optional[str]]:
    head = block[0]
    if not _looks_like_tx_header(head):
        return None, None
    dm = _parse_date(head.cell("tanggal"))
    assert dm is not None
    day, month = dm
    amount = extract_amount(head.cell("mutasi"))
    if amount is None:
        return None, f"unparseable amount: {head.cell('mutasi')!r}"

    tag = head.cell("mutasi").strip().upper().split()
    tx_type: TxType = "DB" if (tag and tag[-1] == "DB") else "CR"
    if tx_type == "CR" and ("DEBIT" in head.cell("keterangan").upper() or " DB" in head.cell("keterangan").upper()):
        tx_type = "DB"

    desc_parts = [head.cell("keterangan")] + [r.cell("keterangan") for r in block[1:] if r.cell("keterangan")]
    keterangan = " | ".join(p for p in desc_parts if p)
    cbg_parts = [r.cell("cbg") for r in block if r.cell("cbg")]
    cbg = " ".join(cbg_parts) or None

    saldo = extract_amount(head.cell("saldo"))
    if saldo is None:
        for r in block[1:]:
            s = extract_amount(r.cell("saldo"))
            if s is not None:
                saldo = s
                break

    return Transaction(
        tanggal=f"{year:04d}-{month:02d}-{day:02d}",
        keterangan=keterangan,
        cbg=cbg,
        amount=amount,
        type=tx_type,
        saldo=saldo,
        page=head.page,
    ), None


# --------------------------- helpers --------------------------------------

def _parse_date(text: str) -> Optional[tuple[int, int]]:
    m = DATE_RE.match(text or "")
    if not m:
        return None
    d, mo = int(m.group(1)), int(m.group(2))
    return (d, mo) if 1 <= d <= 31 and 1 <= mo <= 12 else None


def _first_name_line(page_chunks: list[TextChunk]) -> Optional[str]:
    """Account-holder name from the upper-left address block.

    BCA's header always lays out the address block as:

        KCP <branch name>           <- the branch line (or KCU / KK variants)
        <ACCOUNT HOLDER NAME>       <- the name we want (target)
        <address line 1>
        <address line 2>
        ...
        INDONESIA

    The reliable anchor is the *branch* line at the top, not the variable
    street-name prefix at the bottom. We find KCP/KCU/KK and take the first
    text line directly below it in the left column. If multiple chunks share
    that baseline (long names split into rects), we concatenate left-to-right.
    """
    left = [c for c in page_chunks if c.x0 < 200 and c.text.strip()]
    if not left:
        return None

    # Primary anchor — the BCA branch identifier line.
    branch_y: Optional[float] = None
    for c in sorted(left, key=lambda c: c.y0):
        upper = c.text.strip().upper()
        if upper.startswith(("KCP", "KCU", "KK ", "KANTOR ")):
            branch_y = c.y1
            break

    # Fallback — just below the "REKENING TAHAPAN" title line.
    if branch_y is None:
        title = [c for c in page_chunks if "REKENING TAHAPAN" in c.text.upper()]
        if title:
            branch_y = min(c.y1 for c in title) + 10
        else:
            return None  # nothing recognisable to anchor on

    below = sorted([c for c in left if c.y0 > branch_y], key=lambda c: c.y0)
    if not below:
        return None

    first = below[0]
    same_line = [c for c in below if abs(c.y_center - first.y_center) < 3.0]
    same_line.sort(key=lambda c: c.x0)
    name = " ".join(c.text.strip() for c in same_line).strip()
    return name or None
