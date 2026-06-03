"""Parser for Bank Permata 'Rekening Koran' statements.

Permata's e-statement uses 6 columns with bilingual headers
(Indonesian above English):

    Tgl Trx.       Tgl Valuta     Uraian Trx.       Debet     Kredit      Saldo
    Trx. Date      Val. Date      Trx. Description  Debit     Credit      Balance
    (dd/mm)        (dd/mm)

Key differences from the other supported parsers:

* **Two date columns.** ``Tgl Trx.`` (the transaction date, used by us) and
  ``Tgl Valuta`` (the value date). Both are ``DD/MM``; the year comes from
  the statement's ``Periode Laporan`` header.
* **Indonesian number format** — ``.`` thousands, ``,`` decimal
  (e.g. ``44.752.854,33``). Same as Mandiri; opposite of BCA/BRI. We reuse
  ``_parse_id_amount`` rather than the common helper.
* **Separate Debet / Kredit columns**, like BRI, but no ``0.00`` placeholder
  — the unused column is simply empty. ``type = "DB"`` if Debet has a
  value; ``"CR"`` if Kredit does.
* **Multi-line Uraian.** Each transaction's description typically spans 2–3
  visual lines (counter-party name, channel, time). Following rows without
  a TANGGAL+amount pair belong to the preceding transaction.
* **Much larger page coordinate space** (~2000 pt wide vs ~600 for BCA),
  which is fine — every boundary is computed from header positions, so the
  scale doesn't matter.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..models import AccountHeader, TextChunk, Transaction, TxType
from .common import ParseResult, Row, cluster_lines, join_cell

# Column header tokens — Permata's Indonesian line above the English subheader.
COLUMN_NAMES_ID: tuple[str, ...] = (
    "Tgl Trx.", "Tgl Valuta", "Uraian Trx.", "Debet", "Kredit", "Saldo",
)

# Transaction date pattern in the leftmost column: just DD/MM. Year comes from
# the statement's PERIODE header (e.g. "01 FEBRUARI 2026 - 28 FEBRUARI 2026").
DATE_RE = re.compile(r"^\s*(\d{1,2})/(\d{1,2})\s*$")

# Indonesian month names used in the PERIODE header.
ID_MONTHS = {
    "JANUARI": 1, "FEBRUARI": 2, "MARET": 3, "APRIL": 4, "MEI": 5, "JUNI": 6,
    "JULI": 7, "AGUSTUS": 8, "SEPTEMBER": 9, "OKTOBER": 10, "NOVEMBER": 11, "DESEMBER": 12,
}


@dataclass
class _PermataLayout:
    tanggal: tuple[float, float]    # "Tgl Trx."
    tgl_valuta: tuple[float, float]  # "Tgl Valuta"
    uraian: tuple[float, float]
    debet: tuple[float, float]
    kredit: tuple[float, float]
    saldo: tuple[float, float]


# --------------------------- public entry points --------------------------

def parse_header(chunks: list[TextChunk]) -> AccountHeader:
    page1 = [c for c in chunks if c.page == 1]
    return AccountHeader(
        bank="Permata",
        no_rekening=_permata_field(page1, "No. Rekening"),
        nama=_holder_name(page1),
        periode=_permata_field(page1, "Periode Laporan"),
        mata_uang=_permata_field(page1, "Mata Uang"),
    )


def parse_transactions(chunks: list[TextChunk], header: AccountHeader) -> ParseResult:
    year = _year_from_periode(header.periode)
    transactions: list[Transaction] = []
    parse_warnings: list[str] = []
    balance_warnings: list[str] = []
    last_layout: Optional[_PermataLayout] = None
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

def _detect_column_layout(page_chunks: list[TextChunk]) -> tuple[Optional[_PermataLayout], float]:
    """Find the Indonesian column-header row containing all 6 column labels."""
    candidates = {name: [c for c in page_chunks if c.text.strip() == name]
                  for name in COLUMN_NAMES_ID}
    if not all(candidates.values()):
        return None, 0.0
    for tanggal in candidates["Tgl Trx."]:
        picked: dict[str, TextChunk] = {"Tgl Trx.": tanggal}
        ok = True
        for name in COLUMN_NAMES_ID[1:]:
            # Permata's header chunks share the same baseline within a few pt
            # in the document's coordinate space — keep a generous tolerance
            # since the page scale is much larger than BCA/BRI.
            match = next((c for c in candidates[name] if abs(c.y_center - tanggal.y_center) < 8.0), None)
            if match is None:
                ok = False
                break
            picked[name] = match
        if not ok:
            continue
        layout = _PermataLayout(
            tanggal=(picked["Tgl Trx."].x0, picked["Tgl Trx."].x1),
            tgl_valuta=(picked["Tgl Valuta"].x0, picked["Tgl Valuta"].x1),
            uraian=(picked["Uraian Trx."].x0, picked["Uraian Trx."].x1),
            debet=(picked["Debet"].x0, picked["Debet"].x1),
            kredit=(picked["Kredit"].x0, picked["Kredit"].x1),
            saldo=(picked["Saldo"].x0, picked["Saldo"].x1),
        )
        # Include the English subheader and the (dd/mm) hint row in the
        # header band (~90 pt below the main row in the document's scale).
        header_bottom = max(picked[n].y1 for n in COLUMN_NAMES_ID) + 100.0
        return layout, header_bottom
    return None, 0.0


def _column_boundaries(layout: _PermataLayout) -> tuple[float, float, float, float, float]:
    """X-thresholds calibrated for Permata Rekening Koran.

    * Tgl Trx / Tgl Valuta : midpoint between the two date-header spans
    * Tgl Valuta / Uraian  : midpoint between Tgl Valuta and Uraian headers
    * Uraian / Debet       : Uraian content (counter-party names) extends
                             well past its centered header, so we place the
                             boundary halfway between Uraian's right edge
                             and Debet's left header edge — but biased
                             toward Debet so wide descriptions are still
                             captured as Uraian.
    * Debet / Kredit       : header midpoint
    * Kredit / Saldo       : header midpoint
    """
    return (
        (layout.tanggal[1] + layout.tgl_valuta[0]) / 2,
        (layout.tgl_valuta[1] + layout.uraian[0]) / 2,
        layout.debet[0] - 200.0,
        (layout.debet[1] + layout.kredit[0]) / 2,
        (layout.kredit[1] + layout.saldo[0]) / 2,
    )


def _column_for_x(x: float, layout: _PermataLayout) -> str:
    b1, b2, b3, b4, b5 = _column_boundaries(layout)
    if x < b1: return "tanggal"
    if x < b2: return "tgl_valuta"
    if x < b3: return "uraian"
    if x < b4: return "debet"
    if x < b5: return "kredit"
    return "saldo"


# --------------------------- row / block assembly -------------------------

def _chunks_to_rows(chunks: list[TextChunk], layout: _PermataLayout, page: int) -> list[Row]:
    rows: list[Row] = []
    # Permata's page coordinate space is ~3.5x larger than BCA's, so the
    # default line-clustering tolerance picks up sub-line jitter as separate
    # lines. We bump the tolerance ratio to compensate.
    for line in cluster_lines(chunks, tol_ratio=1.2):
        per_col: dict[str, list[TextChunk]] = {
            n: [] for n in ("tanggal", "tgl_valuta", "uraian", "debet", "kredit", "saldo")
        }
        for c in line:
            per_col[_column_for_x(c.x_center, layout)].append(c)
        rows.append(Row(
            cells={n: join_cell(per_col[n]) for n in per_col},
            page=page,
            y=sum(c.y_center for c in line) / len(line),
        ))
    return rows


def _looks_like_tx_header(row: Row) -> bool:
    """Permata transaction-anchor rows carry a parseable DD/MM date AND a
    parseable amount in either Debet or Kredit."""
    if _parse_date(row.cell("tanggal")) is None:
        return False
    return (_parse_id_amount(row.cell("debet")) is not None
            or _parse_id_amount(row.cell("kredit")) is not None)


def _group_into_blocks(rows: list[Row]) -> list[list[Row]]:
    blocks: list[list[Row]] = []
    for row in rows:
        if _looks_like_tx_header(row):
            blocks.append([row])
        elif blocks:
            blocks[-1].append(row)
        # rows before the first anchored transaction (e.g. ``SALDO AWAL``)
        # are silently dropped — they don't carry both a date and an amount,
        # so they wouldn't parse as transactions anyway.
    return blocks


def _block_to_transaction(block: list[Row], year: int) -> tuple[Optional[Transaction], Optional[str]]:
    head = block[0]
    if not _looks_like_tx_header(head):
        return None, None
    dm = _parse_date(head.cell("tanggal"))
    assert dm is not None
    day, month = dm

    debet = _parse_id_amount(head.cell("debet"))
    kredit = _parse_id_amount(head.cell("kredit"))
    if debet is not None and (kredit is None or kredit == 0):
        amount, tx_type = debet, "DB"
    elif kredit is not None and (debet is None or debet == 0):
        amount, tx_type = kredit, "CR"
    elif debet and kredit:
        amount = max(debet, kredit)
        tx_type = "CR" if kredit >= debet else "DB"
    else:
        return None, "row has neither debet nor kredit"

    desc_parts = [head.cell("uraian")] + [r.cell("uraian") for r in block[1:] if r.cell("uraian")]
    keterangan = " | ".join(p for p in desc_parts if p)

    saldo = _parse_id_amount(head.cell("saldo"))
    if saldo is None:
        for r in block[1:]:
            s = _parse_id_amount(r.cell("saldo"))
            if s is not None:
                saldo = s
                break

    return Transaction(
        tanggal=f"{year:04d}-{month:02d}-{day:02d}",
        keterangan=keterangan,
        cbg=None,
        amount=amount,
        type=tx_type,  # type: ignore[arg-type]
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


def _parse_id_amount(text: str) -> Optional[float]:
    """Indonesian number format: ``.`` thousands, ``,`` decimals.
    Strips a leading ``+`` or ``-`` and returns the absolute value.
    Returns None when the cell has no parseable number.
    """
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


def _permata_field(page_chunks: list[TextChunk], label: str) -> Optional[str]:
    """Find the value chunk immediately to the right of a label on the same row.

    Permata's account-info block is a two-column key-value list where the
    value sits at roughly the same y as the label and ~250-350 pt to the
    right (in this page's large coordinate space).
    """
    labels = [c for c in page_chunks if c.text.strip() == label]
    if not labels:
        return None
    anchor = labels[0]
    # Value chunks on the same baseline, to the right of the label.
    candidates = [c for c in page_chunks
                  if c.x0 > anchor.x1 + 50
                  and abs(c.y_center - anchor.y_center) < 12.0
                  and c.text.strip() not in {":", ""}]
    candidates.sort(key=lambda c: c.x0)
    return candidates[0].text.strip() if candidates else None


def _holder_name(page_chunks: list[TextChunk]) -> Optional[str]:
    """Permata's header puts the holder name on the line below 'Kepada Yth'
    in the left column. Strategy: anchor on 'Kepada Yth', take the first
    left-column chunk below it that's an uppercase string longer than 6
    chars (the address lines tend to start with ``PERUM``, ``JL``, ``BLOK``
    and are also uppercase but the name comes first below the label).
    """
    label = next((c for c in page_chunks if c.text.strip() == "Kepada Yth"), None)
    if label is None:
        return None
    left_below = [c for c in page_chunks
                  if c.x0 < anchor_x_threshold(page_chunks)
                  and c.y0 > label.y1 - 5
                  and c.text.strip()]
    left_below.sort(key=lambda c: c.y0)
    for c in left_below:
        t = c.text.strip()
        # Skip the label echoes ("Account Statement", etc.) and obvious address
        # prefixes — the name line is the first uppercase string with multiple
        # words that doesn't start with one of those.
        if t.upper().startswith(("KEPADA", "ACCOUNT", "STATEMENT", "PERIODE", "TANGGAL", "NO.")):
            continue
        if t.upper().startswith(("JL ", "JL.", "PERUM", "BLOK", "GG ", "RT", "RW",
                                  "KEC ", "KEL ", "KOTA ", "KOMP", "DUSUN", "DESA",
                                  "KAMPUNG", "INDONESIA")):
            continue
        # Name line: predominantly letters/spaces, at least one space (multi-word)
        if " " in t and sum(1 for ch in t if ch.isalpha()) >= 5:
            return t
    return None


def anchor_x_threshold(page_chunks: list[TextChunk]) -> float:
    """The Permata header is split into a left address block (x < ~700 in
    its coordinate space) and a right account-info block (x > ~900). We
    derive the split dynamically by taking the median x0 of all chunks on
    the page — works regardless of page scale.
    """
    xs = sorted(c.x0 for c in page_chunks)
    if not xs:
        return 600.0
    return xs[len(xs) // 2]


def _year_from_periode(periode: Optional[str]) -> int:
    """Pull the first 4-digit year out of strings like
    ``"01 FEBRUARI 2026 - 28 FEBRUARI 2026"``. Falls back to current year.
    """
    if periode:
        for tok in periode.upper().replace("-", " ").split():
            if tok.isdigit() and len(tok) == 4:
                return int(tok)
    from datetime import date
    return date.today().year
