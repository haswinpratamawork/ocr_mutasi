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

from .matcher import match_all_months
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
    """Best-effort YYYY-MM extraction from a slip filename.

    Looks for patterns like ``Feb 2025``, ``Februari_2025``, ``Apr 2025``,
    ``2025-04``. Returns None when nothing matches — those slips will land
    in ``unmatched_slips`` because no month bucket can accept them.
    """
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
) -> MatchResponse:
    """Pair the uploaded slips with the uploaded bank statements."""
    upstream_errors: list[str] = []
    slips: list[ParsedSlip] = []
    credits: list[GajiCredit] = []

    # Step 1 — fan out upstream calls concurrently.
    slip_task = asyncio.create_task(parse_slips(slip_pdfs))
    mut_task = asyncio.create_task(extract_mutations(mutation_pdfs))

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

    # Step 2 — tag with month.
    for s in slips:
        s.month = _slip_month(s)
    for c in credits:
        c.month = _credit_month(c)

    # Step 3 — group by month, intersected. Slips/credits without a month
    # are left out of buckets entirely; they'll surface as unmatched below.
    months = sorted({s.month for s in slips if s.month}
                    | {c.month for c in credits if c.month})
    by_month: dict[str, tuple[list[ParsedSlip], list[GajiCredit]]] = {}
    for m in months:
        m_slips = [s for s in slips if s.month == m]
        m_credits = [c for c in credits if c.month == m]
        if m_slips:  # only run a month bucket that has at least one slip to assign
            by_month[m] = (m_slips, m_credits)

    # Step 4 — concurrent LLM matcher, one call per month bucket.
    if by_month:
        matches, unmatched_slips, unmatched_credits, matcher_errors = await match_all_months(by_month)
    else:
        matches, matcher_errors = [], []
        unmatched_slips = list(slips)
        unmatched_credits = list(credits)

    # Step 5 — anything outside the per-month buckets is unmatched by default.
    slip_ids_in_match = {id(p.slip) for p in matches}
    slip_ids_returned_unmatched = {id(s) for s in unmatched_slips}
    for s in slips:
        if id(s) not in slip_ids_in_match and id(s) not in slip_ids_returned_unmatched:
            unmatched_slips.append(s)

    credit_ids_in_match = {id(p.credit) for p in matches}
    credit_ids_returned_unmatched = {id(c) for c in unmatched_credits}
    for c in credits:
        if id(c) not in credit_ids_in_match and id(c) not in credit_ids_returned_unmatched:
            unmatched_credits.append(c)

    return MatchResponse(
        matches=matches,
        unmatched_slips=unmatched_slips,
        unmatched_credits=unmatched_credits,
        audit=MatchAudit(
            slip_count=len(slips),
            credit_count=len(credits),
            matched_count=len(matches),
            months_processed=sorted(by_month.keys()),
            matcher_errors=matcher_errors,
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
