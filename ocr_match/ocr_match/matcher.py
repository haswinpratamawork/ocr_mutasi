"""Deterministic slip ↔ credit matcher.

Pairing logic (no LLM — exact-amount rule + month-shift fallback):

  For each slip with a known total_paid:
    1. Look for an unused credit whose amount equals slip.total_paid
       (within ``MATCH_AMOUNT_TOLERANCE_RP``, default Rp 1) in month X+1
       — where X is the slip's filename-derived month. This is the
       common Indonesian payroll pattern: a March slip is typically paid
       and shows up in the bank statement in April.
    2. If no X+1 match exists, fall back to month X (same month).
    3. The first match wins; mark that credit as used.

  Each accepted pair records which pattern fired (``"next_month"`` or
  ``"same_month"``) so downstream UIs can show it.

Why no LLM?
  - The user's source data is precise: payroll-side and bank-side amounts
    agree to the rupiah. Fuzzy matching adds noise, not value.
  - The pattern is determined entirely by month-shift and amount equality,
    both of which are deterministic.
  - Without an LLM call, the matcher is O(N+M) instead of O(months) HTTP
    round-trips — pairing 100 slips against 12 monthly statements takes
    microseconds.
"""
from __future__ import annotations

import logging
from collections import defaultdict

from .config import get_settings
from .models import GajiCredit, MatchPair, ParsedSlip

logger = logging.getLogger(__name__)


def _month_plus_one(month: str) -> str | None:
    """``"2025-02"`` → ``"2025-03"``. Returns None if input isn't YYYY-MM."""
    try:
        year_s, mon_s = month.split("-")
        year, mon = int(year_s), int(mon_s)
    except (ValueError, AttributeError):
        return None
    mon += 1
    if mon > 12:
        mon = 1
        year += 1
    return f"{year:04d}-{mon:02d}"


def _credit_day(credit: GajiCredit) -> int:
    """Extract the day-of-month from an ISO ``tanggal``. 0 on parse failure."""
    try:
        return int(credit.tanggal.split("-")[2])
    except (IndexError, ValueError):
        return 0


def match_all(
    slips: list[ParsedSlip],
    credits: list[GajiCredit],
) -> tuple[list[MatchPair], list[ParsedSlip], list[GajiCredit]]:
    """Pair every slip against the credit list, preferring month X+1 over X.

    Returns:
        (matches, unmatched_slips, unmatched_credits).

    The algorithm is greedy: slips are processed in input order; each one
    grabs the first eligible credit it sees. In David's test data this
    gives a unique correct answer because every (month, amount) tuple is
    unique. If collisions arise in larger datasets we may want a smarter
    assignment (Hungarian / institution-aware tie-break) — recorded as
    future work in the spec.
    """
    settings = get_settings()
    tolerance_rp = settings.match_amount_tolerance_rp

    # Index credits by month for O(1) lookups.
    credits_by_month: dict[str, list[tuple[int, GajiCredit]]] = defaultdict(list)
    for idx, c in enumerate(credits):
        if c.month:
            credits_by_month[c.month].append((idx, c))

    matches: list[MatchPair] = []
    used_credit_ids: set[int] = set()
    matched_slip_ids: set[int] = set()

    for sid, slip in enumerate(slips):
        if slip.total_paid is None or slip.month is None:
            logger.info("matcher: slip %d skipped (missing month or total_paid)", sid)
            continue

        target = float(slip.total_paid)
        # Try X+1 first (the common Indonesian payroll-vs-bank pattern), then X.
        candidate_months: list[tuple[str, str]] = []
        next_m = _month_plus_one(slip.month)
        if next_m:
            candidate_months.append((next_m, "next_month"))
        candidate_months.append((slip.month, "same_month"))

        for cand_month, pattern in candidate_months:
            picked: tuple[int, GajiCredit] | None = None
            for cid, credit in credits_by_month.get(cand_month, []):
                if cid in used_credit_ids:
                    continue
                if abs(float(credit.amount) - target) <= tolerance_rp:
                    picked = (cid, credit)
                    break
            if picked is None:
                continue

            cid, credit = picked
            used_credit_ids.add(cid)
            matched_slip_ids.add(sid)

            diff_rp = float(credit.amount) - target
            diff_pct = diff_rp / target if target else 0.0
            day = _credit_day(credit)
            reason = (
                f"Exact-amount match (diff Rp {diff_rp:+.0f}); "
                f"slip month {slip.month} → credit month {cand_month} "
                f"({'X+1 payroll-lag pattern' if pattern == 'next_month' else 'same-month pattern'})"
            )
            matches.append(MatchPair(
                slip=slip,
                credit=credit,
                confidence=1.0,
                reason=reason,
                amount_diff_rp=diff_rp,
                amount_diff_pct=diff_pct,
                days_off=day,
                match_pattern=pattern,
            ))
            break  # this slip is done; move to next slip

    unmatched_slips = [s for i, s in enumerate(slips) if i not in matched_slip_ids]
    unmatched_credits = [c for i, c in enumerate(credits) if i not in used_credit_ids]
    return matches, unmatched_slips, unmatched_credits
