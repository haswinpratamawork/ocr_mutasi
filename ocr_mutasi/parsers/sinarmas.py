"""Parser for Bank Sinarmas 'Tabungan' / 'Account Statement' statements.

Sinarmas's layout is the most unusual of the supported banks:

* **Page-level inversion.** The period-summary rows (``CLOSING BALANCE``,
  ``MOVEMENT TOTALS``) sit ABOVE the transaction body. The column headers
  sit BELOW it. The account-info footer (period, name, account no, currency,
  category) sits at the very bottom of page 1. So when reading the page
  top-to-bottom in y-coords, you see: summary → reverse-chrono transactions
  → opening balance → column headers → account info.

* **Reverse-chronological order WITHIN the body.** The most recent
  transaction is at the top; the oldest (just above the column header) is
  at the bottom, followed by a ``BALANCE PERIOD START`` opening-balance
  row. The parser collects rows in source order then REVERSES the list so
  the response stays forward-chronological like every other bank.

* **Block layout: anchor at the BOTTOM.** Each transaction's anchor row
  (date + amount + balance) sits at the LOWEST y of its visual block.
  Continuation lines (counterparty bank, counterparty account number,
  QR-code merchant string, branch suffix) sit ABOVE the anchor. We collect
  "pending" continuation lines as we descend and attach them to the next
  anchor we hit below.

* **Bilingual headers.** English on the upper line, Indonesian below.
  Column labels: ``Date / Tanggal`` / ``Description / Keterangan`` /
  ``Detail`` (same word in both) / ``Debit / Debet`` / ``Credit / Kredit``
  / ``Balance / Saldo``.

* **English number format** (``,`` thousands, ``.`` decimal) — same as
  BCA/BRI; opposite of Mandiri/Permata. Uses the shared ``extract_amount``
  helper after a no-space concat of amount-column chunks.

* **Leading-digit chunk split.** Long debit amounts can be rendered as a
  single leading digit chunk plus the rest (e.g. a ``'9'`` chunk followed
  by ``'9,000,000.00'`` for a 99-million debit). The parser concatenates
  amount-column chunks *without* a separator before parsing — letter splits
  in description columns are handled differently (see below).

* **Per-letter word splits.** pypdfium often renders ``Transaction`` as
  ``T`` + ``ransaction`` (or ``JAKARTA`` as ``JAKAR`` + ``TA`` + ``TA``).
  ``_heal_letter_splits`` patches the common cases in description / detail
  cells so the LLM classifier sees a clean string.

* **Two-chunk date.** ``Date`` column carries the day digit (``27``) in one
  chunk and ``Apr 2026`` (or Indonesian month) in another, both
  left-aligned at the same baseline. They join cleanly to ``27 Apr 2026``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date as _date
from typing import Optional

from ..models import AccountHeader, TextChunk, Transaction, TxType
from .common import ParseResult, Row, cluster_lines

# Bilingual column headers. English line is above Indonesian; either anchors
# detection (we try EN first because its labels — ``Date`` / ``Description`` /
# ``Balance`` — are unique in this layout).
COLUMNS_EN: tuple[str, ...] = ("Date", "Description", "Detail", "Debit", "Credit", "Balance")
COLUMNS_ID: tuple[str, ...] = ("Tanggal", "Keterangan", "Detail", "Debet", "Kredit", "Saldo")

# Anchor-row date format: ``<day> <Mon> <YYYY>``. Reassembled from a ``"27"`` +
# ``"Apr 2026"`` chunk pair after column bucketing.
DATE_RE = re.compile(r"(\d{1,2})\s+([A-Za-z]{3})\s+(\d{4})")

EN_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    # Indonesian three-letter variants used by some Sinarmas product layouts.
    "MEI": 5, "AGT": 8, "AGU": 8, "OKT": 10, "DES": 12,
}

# Rows whose description column matches one of these fragments are page-level
# summary rows, not transactions — they get skipped during block grouping so
# their balances / totals don't leak into the next anchor's description. We
# match against fragments (not full phrases) because Sinarmas's pypdfium
# rendering can split / phantom-duplicate letters inside ``MOVEMENT TOTALS``
# (rendered as ``MOVEMENT T OTA TALS``) and ``BALANCE PERIOD`` (rendered as
# ``BALANCE AT AT PERIOD``). Each fragment below is unique to the summary
# rows — none appears in any real transaction's description column.
SUMMARY_FRAGMENTS = ("MOVEMENT", "CLOSING", "PERIOD")


@dataclass
class _SinarmasLayout:
    date: tuple[float, float]
    description: tuple[float, float]
    detail: tuple[float, float]
    debit: tuple[float, float]
    credit: tuple[float, float]
    balance: tuple[float, float]


# --------------------------- public entry points --------------------------

def parse_header(chunks: list[TextChunk]) -> AccountHeader:
    page1 = [c for c in chunks if c.page == 1]
    return AccountHeader(
        bank="Sinarmas",
        no_rekening=_sinarmas_field(page1, ["No. Rekening", "Account No"]),
        nama=_sinarmas_name(page1),
        periode=_sinarmas_field(page1, ["Tanggal", "Date"]),
        mata_uang=_sinarmas_field(page1, ["Mata Uang", "Currency"]),
    )


def parse_transactions(chunks: list[TextChunk], header: AccountHeader) -> ParseResult:
    year = _year_from_periode(header.periode)
    transactions: list[Transaction] = []
    parse_warnings: list[str] = []
    balance_warnings: list[str] = []
    last_layout: Optional[_SinarmasLayout] = None

    by_page: dict[int, list[TextChunk]] = {}
    for c in chunks:
        by_page.setdefault(c.page, []).append(c)

    for page in sorted(by_page):
        page_chunks = by_page[page]
        layout, header_top = _detect_column_layout(page_chunks)
        if layout is None:
            if last_layout is None:
                continue  # no header row anywhere yet — page might be a cover
            layout, header_top = last_layout, 0.0
        else:
            last_layout = layout

        # Body sits ABOVE the column header (higher y). header_top is the y
        # value at which the body begins. Pages without a header (= 0.0)
        # use the whole page as body, relying on summary-row exclusion to
        # discard cover content.
        body_chunks = [c for c in page_chunks if c.y0 > header_top]
        rows = _chunks_to_rows(body_chunks, layout, page)
        # Walk top-to-bottom (descending y) — anchors emerge at the bottom of
        # each visual block; continuation lines accumulate above them.
        rows.sort(key=lambda r: -r.y)

        page_txs, warnings = _rows_to_transactions(rows, year, page)
        parse_warnings.extend(warnings)
        # Forward-chronological for the response.
        page_txs.reverse()
        transactions.extend(page_txs)

    # Running-balance check across the forward-chrono sequence.
    running: Optional[float] = None
    for tx in transactions:
        delta = tx.amount if tx.type == "CR" else -tx.amount
        if running is None:
            if tx.saldo is not None:
                running = tx.saldo
        else:
            running += delta
            if tx.saldo is not None and abs(running - tx.saldo) > 0.5:
                balance_warnings.append(
                    f"p{tx.page} {tx.tanggal}: running {running:.2f} vs printed {tx.saldo:.2f}"
                )
                running = tx.saldo

    return ParseResult(transactions, parse_warnings, balance_warnings)


# --------------------------- column-layout detection ----------------------

def _detect_column_layout(page_chunks: list[TextChunk]) -> tuple[Optional[_SinarmasLayout], float]:
    """Find the bilingual header row. Prefers the English line; falls back to
    Indonesian on continuation pages where only the ID line might render."""
    for names in (COLUMNS_EN, COLUMNS_ID):
        candidates = {n: [c for c in page_chunks if c.text.strip() == n] for n in names}
        if not all(candidates.values()):
            continue
        for anchor in candidates[names[0]]:
            picked: dict[str, TextChunk] = {names[0]: anchor}
            ok = True
            for n in names[1:]:
                match = next((c for c in candidates[n]
                              if abs(c.y_center - anchor.y_center) < 4.0), None)
                if match is None:
                    ok = False
                    break
                picked[n] = match
            if not ok:
                continue
            layout = _SinarmasLayout(
                date=(picked[names[0]].x0, picked[names[0]].x1),
                description=(picked[names[1]].x0, picked[names[1]].x1),
                detail=(picked[names[2]].x0, picked[names[2]].x1),
                debit=(picked[names[3]].x0, picked[names[3]].x1),
                credit=(picked[names[4]].x0, picked[names[4]].x1),
                balance=(picked[names[5]].x0, picked[names[5]].x1),
            )
            # The English header sits ~7 pt above the Indonesian — body begins
            # ~8 pt above the EN row, so use the EN row's top edge as the cut.
            header_top = max(picked[n].y1 for n in names) + 8.0
            return layout, header_top
    return None, 0.0


def _column_boundaries(layout: _SinarmasLayout) -> tuple[float, float, float, float, float]:
    """X-thresholds for the six columns.

    The Date column is unusually narrow (header at x≈35-49) while its content
    spans x≈24-60 (day digit + ``Mon YYYY``), and the Description column's
    content starts at x≈74 — well to the LEFT of its header at x≈123. We
    therefore place the date/description boundary just past the Date header's
    right edge (with a 12 pt slack) instead of using the header midpoint, which
    would catch ``Sales`` (xc≈82) on the wrong side. Every other boundary uses
    the standard header midpoint.
    """
    return (
        layout.date[1] + 12.0,
        (layout.description[1] + layout.detail[0]) / 2,
        (layout.detail[1] + layout.debit[0]) / 2,
        (layout.debit[1] + layout.credit[0]) / 2,
        (layout.credit[1] + layout.balance[0]) / 2,
    )


def _column_for_x(x: float, layout: _SinarmasLayout) -> str:
    b1, b2, b3, b4, b5 = _column_boundaries(layout)
    if x < b1: return "date"
    if x < b2: return "description"
    if x < b3: return "detail"
    if x < b4: return "debit"
    if x < b5: return "credit"
    return "balance"


# --------------------------- row / block assembly -------------------------

def _chunks_to_rows(chunks: list[TextChunk], layout: _SinarmasLayout, page: int) -> list[Row]:
    """Cluster chunks into visual lines and bucket each line into the six
    columns. Description / detail cells use space-joined text (and a
    letter-split healer); amount cells (debit / credit / balance) use a
    NO-SPACE concat so leading-digit splits join cleanly."""
    rows: list[Row] = []
    for line in cluster_lines(chunks):
        per_col: dict[str, list[TextChunk]] = {
            n: [] for n in ("date", "description", "detail", "debit", "credit", "balance")
        }
        for c in line:
            per_col[_column_for_x(c.x_center, layout)].append(c)
        cells = {
            "date":        _text_join(per_col["date"]),
            "description": _heal_letter_splits(_text_join(per_col["description"])),
            "detail":      _heal_letter_splits(_text_join(per_col["detail"])),
            "debit":       _amount_join(per_col["debit"]),
            "credit":      _amount_join(per_col["credit"]),
            "balance":     _amount_join(per_col["balance"]),
        }
        rows.append(Row(
            cells=cells,
            page=page,
            y=sum(c.y_center for c in line) / len(line),
        ))
    return rows


def _rows_to_transactions(
    rows: list[Row],
    year_hint: int,
    page: int,
) -> tuple[list[Transaction], list[str]]:
    """Walk rows top-to-bottom (already sorted). Accumulate continuation rows
    in `pending` until an anchor row appears; flush them onto the anchor."""
    transactions: list[Transaction] = []
    warnings: list[str] = []
    pending_desc: list[str] = []
    pending_detail: list[str] = []

    for row in rows:
        if _is_summary_row(row):
            # Period-summary rows (CLOSING BALANCE / MOVEMENT TOTALS / BALANCE
            # PERIOD START) — drop accumulated pending so they don't bleed
            # across the summary boundary.
            pending_desc.clear()
            pending_detail.clear()
            continue

        date_parts = _parse_date(row.cell("date"))
        debit = _parse_amount(row.cell("debit"))
        credit = _parse_amount(row.cell("credit"))

        if date_parts is not None and (debit is not None or credit is not None):
            # ANCHOR — finalize a transaction.
            day, month, parsed_year = date_parts
            year = parsed_year or year_hint
            if debit is not None and (credit is None or credit == 0):
                amount, tx_type = debit, "DB"
            elif credit is not None and (debit is None or debit == 0):
                amount, tx_type = credit, "CR"
            else:  # both present — shouldn't happen on Sinarmas, but be safe
                if (credit or 0) >= (debit or 0):
                    amount, tx_type = credit, "CR"
                else:
                    amount, tx_type = debit, "DB"

            # Description = anchor's own description column, prefixed by the
            # continuation lines we collected above (in source order: top to
            # bottom = same direction we walked them).
            desc_chunks = [s for s in pending_desc if s] + [row.cell("description")]
            desc_chunks = [s for s in desc_chunks if s]
            detail_chunks = [s for s in pending_detail if s] + [row.cell("detail")]
            detail_chunks = [s for s in detail_chunks if s]
            # Combine description and detail with a separator — detail carries
            # the counterparty (bank / account / name) which is meaningful for
            # downstream classification.
            full_desc_parts = desc_chunks + detail_chunks
            keterangan = " | ".join(full_desc_parts)

            saldo = _parse_amount(row.cell("balance"))

            transactions.append(Transaction(
                tanggal=f"{year:04d}-{month:02d}-{day:02d}",
                keterangan=keterangan,
                cbg=None,
                amount=amount,
                type=tx_type,  # type: ignore[arg-type]
                saldo=saldo,
                page=page,
            ))
            pending_desc.clear()
            pending_detail.clear()
        else:
            # Continuation. Description / detail cells are accumulated; rows
            # carrying only a balance figure (none seen so far but possible
            # on multi-page statements) get dropped.
            if row.cell("description"):
                pending_desc.append(row.cell("description"))
            if row.cell("detail"):
                pending_detail.append(row.cell("detail"))

    return transactions, warnings


def _is_summary_row(row: Row) -> bool:
    """True if the description column matches a page-level summary marker.

    The check is done on the WHITESPACE-STRIPPED uppercase form so we catch
    rows whose label is fragmented by pypdfium (``MOVEMENT T OTA TALS`` /
    ``BALANCE AT AT PERIOD``). Each fragment in ``SUMMARY_FRAGMENTS`` is
    unique to summary rows.
    """
    desc = row.cell("description").upper().replace(" ", "")
    return any(frag in desc for frag in SUMMARY_FRAGMENTS)


# --------------------------- cell-text helpers ----------------------------

def _text_join(chunks: list[TextChunk]) -> str:
    """Space-joined left-to-right concat — for date / description / detail."""
    if not chunks:
        return ""
    return " ".join(c.text for c in sorted(chunks, key=lambda c: c.x0)).strip()


def _amount_join(chunks: list[TextChunk]) -> str:
    """No-space concat for amount columns — collapses a leading-digit-split
    chunk pair (e.g. ``'9' '9,000,000.00'`` → ``'99,000,000.00'``) while
    leaving Indonesian/English number formats parseable by ``_parse_amount``."""
    if not chunks:
        return ""
    return "".join(c.text for c in sorted(chunks, key=lambda c: c.x0)).strip()


_LETTER_SPLIT_LOWER = re.compile(r"\b([A-Z]) ([a-z]{2,})")
_LETTER_SPLIT_UPPER_LONG = re.compile(r"\b([A-Z]{2,}) ([A-Z]{1,3})\b")
_LETTER_SPLIT_UPPER_SHORT = re.compile(r"\b([A-Z]) ([A-Z]{1,3})\b")


def _dedupe_phantom_tokens(text: str) -> str:
    """Drop tokens that look like pypdfium phantom-renders of an adjacent token.

    Three patterns are handled:

    * **Prefix phantom** — a short token (≤3 chars) whose chars also start the
      next token. ``"Ta Tax"`` → ``"Tax"``.
    * **Suffix phantom** — a short token (≤4 chars) that exactly matches the
      tail of the previous token. ``"PT. T."`` → ``"PT."``,
      ``"JAKARTA TA"`` → ``"JAKARTA"``.
    * **Adjacent duplicate** — two consecutive identical short tokens.
      ``"TA TA"`` → ``"TA"``.

    The healer is conservative — it only fires for tokens up to 4 characters,
    so genuine two-word phrases like ``"PT BUDI SANTOSO"`` are unaffected.
    """
    tokens = text.split()
    if len(tokens) < 2:
        return text
    out: list[str] = []
    for i, t in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        # Prefix phantom: short token whose chars start the next token.
        if 1 <= len(t) <= 3 and nxt and nxt.startswith(t):
            continue
        # Suffix phantom: short token matching the tail of the previous one
        # (and not just a single letter that happens to coincide — require
        # the previous token to be ≥2 chars longer).
        if out and 1 <= len(t) <= 4 and out[-1].endswith(t) and len(out[-1]) >= len(t) + 1:
            continue
        # Adjacent duplicate of the previous short token.
        if out and out[-1] == t and len(t) <= 4:
            continue
        out.append(t)
    return " ".join(out)


def _heal_letter_splits(text: str) -> str:
    """Patch the most common pypdfium per-letter splits in description text.

    Examples:
      - ``Sales T ransaction`` → ``Sales Transaction``
      - ``Auto T ransfer Credit`` → ``Auto Transfer Credit``
      - ``Ta Tax Amount Due`` → ``Tax Amount Due``
      - ``JAKAR TA TA`` → ``JAKARTA``
      - ``PT. T. BANK JASA`` → ``PT. BANK JASA``
      - ``MOVEMENT T OTA TALS`` → ``MOVEMENT TOTA TALS`` (the underlying
        ``TOTALS`` rendering is too fragmented to fully reconstruct, but
        summary-row detection runs on a whitespace-stripped check that still
        recognises it correctly).

    Order matters: dedupe phantoms FIRST so we don't merge a phantom into a
    real word ("Ta" + "Tax" → "TaTax" if we let the lowercase healer run
    first); THEN the per-letter regex healers.
    """
    if not text:
        return text
    text = _dedupe_phantom_tokens(text)
    prev = None
    out = text
    # Iterate until stable — collapses chains like ``JAKAR TA TA``.
    while out != prev:
        prev = out
        out = _LETTER_SPLIT_LOWER.sub(r"\1\2", out)
        out = _LETTER_SPLIT_UPPER_LONG.sub(r"\1\2", out)
        out = _LETTER_SPLIT_UPPER_SHORT.sub(r"\1\2", out)
        # Re-run dedupe in case the regex merges revealed new phantoms.
        out = _dedupe_phantom_tokens(out)
    return re.sub(r"\s{2,}", " ", out).strip()


# --------------------------- value parsers --------------------------------

def _parse_date(text: str) -> Optional[tuple[int, int, Optional[int]]]:
    """``"27 Apr 2026"`` → ``(27, 4, 2026)``. Returns ``None`` when the cell
    doesn't carry a parseable date — that's the signal for "continuation row"."""
    if not text:
        return None
    m = DATE_RE.search(text)
    if not m:
        return None
    day = int(m.group(1))
    mon_str = m.group(2).upper()
    year = int(m.group(3))
    month = EN_MONTHS.get(mon_str)
    if month is None or not (1 <= day <= 31):
        return None
    return day, month, year


def _parse_amount(text: str) -> Optional[float]:
    """Pull the first parseable number out of an amount cell.

    Sinarmas uses English number format (``,`` thousands, ``.`` decimal).
    Cells are no-space concats of one or more chunks; after stripping any
    leading sign / currency marker we accept the longest digit run.
    """
    if not text:
        return None
    s = text.strip().lstrip("+-Rp ").strip()
    if not s:
        return None
    # First contiguous run of [digits, commas, optional `.dd` decimal].
    m = re.search(r"\d[\d,]*(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group().replace(",", ""))
    except ValueError:
        return None


# --------------------------- header-field helpers -------------------------

def _sinarmas_field(page_chunks: list[TextChunk], labels: list[str]) -> Optional[str]:
    """Find the value chunk to the right of any of the candidate labels on the
    same baseline. Tries each label in order (Indonesian preferred over
    English so the response is consistent across pages).

    Skips the tiny render-artifact chunks (``'Ta'``, ``':'``, single chars)
    that pypdfium emits next to some Sinarmas labels — we pick the longest
    text chunk on the same row to the right of the label.
    """
    for label in labels:
        anchors = [c for c in page_chunks if c.text.strip() == label]
        if not anchors:
            continue
        anchor = anchors[0]
        candidates = [c for c in page_chunks
                      if c.x0 > anchor.x1 + 5
                      and abs(c.y_center - anchor.y_center) < 4.0
                      and c.text.strip() not in {":", ""}
                      and len(c.text.strip()) >= 3]
        if candidates:
            return max(candidates, key=lambda c: len(c.text.strip())).text.strip()
    return None


def _sinarmas_name(page_chunks: list[TextChunk]) -> Optional[str]:
    """Holder name — sits in the left column, slightly below the right-column
    ``No. Rekening`` label and above the first address line.

    Strategy: anchor on ``No. Rekening``, scan left-column chunks within
    ~30 pt below it for the first uppercase multi-word string that doesn't
    look like an address line. The Indonesian label is preferred; we fall
    back to ``Account No`` if it isn't found.
    """
    anchor = None
    for label in ("No. Rekening", "Account No"):
        hits = [c for c in page_chunks if c.text.strip() == label]
        if hits:
            anchor = hits[0]
            break
    if anchor is None:
        return None

    # Sinarmas left column: x < ~200 in this page's coordinate space. Make it
    # robust to scale by deriving from page width.
    page_width = max((c.x1 for c in page_chunks), default=600.0)
    left_cutoff = page_width * 0.40  # leaves room for typical name length
    candidates = [c for c in page_chunks
                  if c.x0 < left_cutoff
                  and anchor.y0 - 30 < c.y_center < anchor.y_center
                  and c.text.strip()]
    candidates.sort(key=lambda c: -c.y_center)  # closest-below first

    for c in candidates:
        t = c.text.strip()
        if not _looks_like_holder_name(t):
            continue
        return t
    return None


# Common Indonesian address-line lead-ins that we explicitly DO NOT want to
# pick up as the holder name.
_ADDRESS_PREFIXES = (
    "JL ", "JL.", "JALAN ",
    "BLOK ", "GG ", "RT ", "RW ", "RT/", "RW/",
    "KEC ", "KEL ", "KOTA ", "KOMP", "RUKO",
    "TAMAN ", "PERUM", "DUSUN", "DESA",
    "KAMPUNG ", "INDONESIA",
)


def _looks_like_holder_name(text: str) -> bool:
    """True for an Indonesian-style all-caps holder name (e.g.
    ``"BUDI SANTOSO"``), false for address lines and junk fragments. Holder
    names are all-caps multi-word strings of letters (with possible periods
    / commas) ≥ 5 letters total."""
    if not text.isupper():
        return False
    if any(text.startswith(p) for p in _ADDRESS_PREFIXES):
        return False
    if not all(ch.isalpha() or ch.isspace() or ch in ".,'" for ch in text):
        return False
    if " " not in text:
        return False
    letters = sum(1 for ch in text if ch.isalpha())
    return letters >= 5


def _year_from_periode(periode: Optional[str]) -> int:
    """First 4-digit year token wins. Falls back to today's year."""
    if periode:
        for m in re.findall(r"\d{4}", periode):
            return int(m)
    return _date.today().year
