"""End-to-end orchestrator.

Single public function ``run(slip_pdfs, mutation_pdfs)``:

  1. Concurrently call ``upstream.parse_slips`` and ``upstream.extract_mutations``.
  2. Derive each item's month (YYYY-MM) — slip from filename, credit from tanggal.
  3. Group both by month.
  4. Hand each month bucket to ``matcher.match_month`` (running in parallel).
  5. Assemble the final ``MatchResponse``.

All filesystem and PDF I/O is delegated to the upstream services. This file
only juggles already-parsed JSON.
"""
from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from typing import Optional

from .matcher import match_all
from .models import GajiCredit, MatchAudit, MatchResponse, ParsedSlip
from .upstream import (
    UpstreamHttpError,
    UpstreamUnreachableError,
    extract_mutations,
    parse_slips,
)

logger = logging.getLogger(__name__)


# --------------------------- month derivation -----------------------------

# English short months and Indonesian variants seen in slip filenames.
_MONTHS = {
    "JAN": 1, "JANUARI": 1,
    "FEB": 2, "FEBRUARI": 2,
    "MAR": 3, "MARET": 3, "MARCH": 3,
    "APR": 4, "APRIL": 4,
    "MAY": 5, "MEI": 5,
    "JUN": 6, "JUNI": 6, "JUNE": 6,
    "JUL": 7, "JULI": 7, "JULY": 7,
    "AUG": 8, "AGT": 8, "AGUSTUS": 8, "AUGUST": 8,
    "SEP": 9, "SEPT": 9, "SEPTEMBER": 9,
    "OCT": 10, "OKT": 10, "OKTOBER": 10, "OCTOBER": 10,
    "NOV": 11, "NOVEMBER": 11,
    "DEC": 12, "DES": 12, "DESEMBER": 12, "DECEMBER": 12,
}

_MONTH_NAME_RE = re.compile(
    r"\b(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\b[\s_/-]*(\d{4})",
    re.IGNORECASE,
)


def _slip_month(slip: ParsedSlip) -> Optional[str]:
    """Best-effort YYYY-MM extraction for a slip.

    Source-of-truth ordering:
      1. ``slip.period`` (emitted by ocr_slip from the slip's ``Period:`` /
         ``Periode:`` line) — most reliable because it comes from the slip's
         own content.
      2. ``slip.source_file`` — patterns like ``Feb 2025``, ``Februari_2025``,
         ``Apr 2025``, or ``2025-04`` anywhere in the filename.

    Returns ``None`` when nothing matches — the matcher then falls back to
    a cross-month amount-only search (see ``matcher.match_all``).
    """
    if slip.period:
        return slip.period
    name = slip.source_file or ""
    m = _MONTH_NAME_RE.search(name)
    if m:
        mon = _MONTHS[m.group(1).upper()]
        year = int(m.group(2))
        return f"{year:04d}-{mon:02d}"
    # Fallback: explicit "YYYY-MM" anywhere in the filename
    m2 = re.search(r"\b(\d{4})[-_/](\d{2})\b", name)
    if m2:
        return f"{int(m2.group(1)):04d}-{int(m2.group(2)):02d}"
    return None


def _credit_month(credit: GajiCredit) -> Optional[str]:
    """Credits use ISO ``YYYY-MM-DD`` for ``tanggal``; just slice off the day."""
    if credit.tanggal and len(credit.tanggal) >= 7:
        return credit.tanggal[:7]
    return None


# --------------------------- public entry ---------------------------------

async def run(
    slip_pdfs: list[tuple[str, bytes]],
    mutation_pdfs: list[tuple[str, bytes]],
    *,
    slip_password: str | None = None,
    mutation_password: str | None = None,
) -> MatchResponse:
    """Pair the uploaded slips with the uploaded bank statements.

    ``slip_password`` and ``mutation_password`` are forwarded independently
    to the two upstream services. The two file groups commonly use different
    encryption (slip = employee ID, bank statement = account-no last 6).
    """
    upstream_errors: list[str] = []
    slips: list[ParsedSlip] = []
    credits: list[GajiCredit] = []

    # Step 1 — fan out upstream calls concurrently.
    slip_task = asyncio.create_task(parse_slips(slip_pdfs, password=slip_password))
    mut_task = asyncio.create_task(extract_mutations(mutation_pdfs, password=mutation_password))

    try:
        slips = await slip_task
    except (UpstreamUnreachableError, UpstreamHttpError) as exc:
        upstream_errors.append(f"ocr_slip: {exc}")
        mut_task.cancel()
        return _empty_response([], [], upstream_errors)

    try:
        credits = await mut_task
    except (UpstreamUnreachableError, UpstreamHttpError) as exc:
        upstream_errors.append(f"ocr_mutasi: {exc}")
        return _empty_response(slips, [], upstream_errors)

    # Step 2 — tag each item with its month (YYYY-MM).
    for s in slips:
        s.month = _slip_month(s)
    for c in credits:
        c.month = _credit_month(c)

    # Step 3 — run the deterministic matcher across all slips and credits.
    # The matcher itself handles the X+1 / X month-shift logic and exact-amount
    # rule — no per-month grouping needed here.
    matches, unmatched_slips, unmatched_credits = match_all(slips, credits)

    # ``months_processed`` is now informational: every distinct slip month
    # (and, for visibility, the next-month buckets we looked into).
    slip_months = {s.month for s in slips if s.month}
    next_months = set()
    for m in slip_months:
        try:
            y, mo = m.split("-")
            mo_i = int(mo) + 1
            yr_i = int(y)
            if mo_i > 12:
                mo_i, yr_i = 1, yr_i + 1
            next_months.add(f"{yr_i:04d}-{mo_i:02d}")
        except (ValueError, AttributeError):
            pass
    months_processed = sorted(slip_months | next_months)

    return MatchResponse(
        matches=matches,
        unmatched_slips=unmatched_slips,
        unmatched_credits=unmatched_credits,
        audit=MatchAudit(
            slip_count=len(slips),
            credit_count=len(credits),
            matched_count=len(matches),
            months_processed=months_processed,
            matcher_errors=[],  # deterministic matcher can't fail
            upstream_errors=upstream_errors,
        ),
    )


def _empty_response(
    slips: list[ParsedSlip],
    credits: list[GajiCredit],
    upstream_errors: list[str],
) -> MatchResponse:
    return MatchResponse(
        matches=[],
        unmatched_slips=list(slips),
        unmatched_credits=list(credits),
        audit=MatchAudit(
            slip_count=len(slips),
            credit_count=len(credits),
            matched_count=0,
            months_processed=[],
            matcher_errors=[],
            upstream_errors=upstream_errors,
        ),
    )
