"""Deterministic slip ↔ credit matcher.

Pairing logic (no LLM — exact-amount rule + ordered month search):

  For each slip with a known total_paid, we try candidate months in this
  preference order and take the first credit whose amount equals
  slip.total_paid (within ``MATCH_AMOUNT_TOLERANCE_RP``, default Rp 1):

    1. **X+1**  — the common Indonesian payroll pattern: a March slip
                  shows up in the bank statement in April.
    2. **X**    — same-month payroll (some companies pay in-month).
    3. **X+2**, **X+3**  — delayed payroll. Late-disbursing companies are
                          common: a February slip can land in April when
                          the employer's payroll cycle runs late.
    4. Any other month in chronological order — pure exact-amount match.

  When ``slip.month`` is ``None`` (the slip carries no period info), we
  skip steps 1–3 and go straight to step 4 — a global exact-amount search
  across every credit. This is safe because payroll-sized amounts are
  precise to the rupiah and rarely collide with non-payroll credits, AND
  because the upstream classifier has already filtered to Gaji-only
  credits before the matcher sees them.

  Each accepted pair records which pattern fired (``"next_month"``,
  ``"same_month"``, ``"future_month"``, or ``"amount_only"``) so downstream
  UIs can surface why the pairing was made.

Why no LLM?
  - The user's source data is precise: payroll-side and bank-side amounts
    agree to the rupiah. Fuzzy matching adds noise, not value.
  - Exact-amount equality + month preference is fully deterministic.
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


def _month_shift(month: str, offset: int) -> str | None:
    """``"2025-02"`` + 1 → ``"2025-03"``. Returns None if input isn't YYYY-MM."""
    try:
        year_s, mon_s = month.split("-")
        year, mon = int(year_s), int(mon_s)
    except (ValueError, AttributeError):
        return None
    mon += offset
    while mon > 12:
        mon -= 12
        year += 1
    while mon < 1:
        mon += 12
        year -= 1
    return f"{year:04d}-{mon:02d}"


# Max payroll-lag offset we'll search after the slip's stated period before
# falling back to the all-months sweep. Three months covers the realistic
# delayed-payroll cases (Indonesian companies occasionally run 1–2 months
# late at year boundaries or after holiday seasons).
_MAX_FORWARD_LAG = 3


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

    # Stable chronological ordering of every credit month — used both for the
    # forward-lag sweep beyond X+3 and for the no-slip-month fallback.
    all_months_sorted: list[str] = sorted(credits_by_month.keys())

    matches: list[MatchPair] = []
    used_credit_ids: set[int] = set()
    matched_slip_ids: set[int] = set()

    for sid, slip in enumerate(slips):
        if slip.total_paid is None:
            logger.info("matcher: slip %d skipped (missing total_paid)", sid)
            continue

        target = float(slip.total_paid)
        candidate_months = _candidate_months(slip.month, all_months_sorted)

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
            reason = _pattern_reason(slip.month, cand_month, pattern, diff_rp)
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


def _candidate_months(
    slip_month: str | None,
    all_months_sorted: list[str],
) -> list[tuple[str, str]]:
    """Ordered (month, pattern) list to search for a slip.

    With a known ``slip_month`` (X), preference order is:
      X+1 → X → X+2 → X+3 → every other month chronologically (last resort).

    With ``slip_month=None``, we go straight to a chronological sweep of every
    credit month and tag the matches as ``amount_only``.
    """
    if slip_month is None:
        return [(m, "amount_only") for m in all_months_sorted]

    ordered: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(m: str | None, pattern: str) -> None:
        if m and m not in seen:
            ordered.append((m, pattern))
            seen.add(m)

    next_m = _month_shift(slip_month, +1)
    add(next_m, "next_month")
    add(slip_month, "same_month")
    add(_month_shift(slip_month, +2), "future_month")
    add(_month_shift(slip_month, +3), "future_month")
    # Final sweep: any credit month we haven't tried yet.
    for m in all_months_sorted:
        add(m, "amount_only")
    return ordered


def _pattern_reason(
    slip_month: str | None,
    credit_month: str,
    pattern: str,
    diff_rp: float,
) -> str:
    """Human-readable reason string for the matched pair, naming the pattern."""
    diff_str = f"diff Rp {diff_rp:+.0f}"
    if pattern == "next_month":
        label = "X+1 payroll-lag pattern"
    elif pattern == "same_month":
        label = "same-month pattern"
    elif pattern == "future_month":
        label = "delayed-payroll pattern (X+2 or X+3)"
    else:  # amount_only
        label = "exact-amount match (no month preference)"
    if slip_month:
        return f"Exact-amount match ({diff_str}); slip month {slip_month} → credit month {credit_month} ({label})"
    return f"Exact-amount match ({diff_str}); slip has no period → searched all credit months ({label}, hit {credit_month})"
