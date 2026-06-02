"""Per-month LLM matcher.

For each month we send the LLM that month's slips and Gaji credits and ask
it to assign each slip to at most one credit (or none). Cross-month
combinations are by construction invalid and excluded — see spec §6.3.

The prompt is explicit, rule-based, and modelled after ocr_mutasi's
classifier prompt: hard rules first, soft signals second, structured-output
JSON schema strictly enforced. Same audit story (the LLM's `reason` field
names the rule/signal that drove its choice).
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from openai import APIError, APITimeoutError, AsyncAzureOpenAI

from .config import get_settings
from .models import GajiCredit, MatchPair, ParsedSlip

logger = logging.getLogger(__name__)

# ----------------------------- prompt --------------------------------------

SYSTEM_PROMPT = """You pair Indonesian salary slips with bank-credit rows \
that paid them. You receive ONE month's worth at a time. Output JSON pairing \
each slip with either ONE credit or null.

## Hard rules (a pairing is invalid if any fails)

1. The credit's month MUST equal the slip's month.
2. |credit.amount − slip.total_paid| / slip.total_paid ≤ {tolerance:.2f}
   (covers small tax/fee discrepancies; default 15%).
3. Each credit may be assigned to AT MOST ONE slip.

## Soft signals (use to disambiguate when multiple credits pass the hard rules)

* Institution-name fuzzy match. Common Indonesian abbreviations:
  - "Alsut"        ≡ "Alam Sutera"
  - "Bintaro"      → often "BSD" (Bumi Serpong Damai, neighbouring area)
  - "Jaktim"       ≡ "Jakarta Timur"
  - "Jakbar"       ≡ "Jakarta Barat"
  - "Jaksel"       ≡ "Jakarta Selatan"
  - "Jakut"        ≡ "Jakarta Utara"
  - "PT <X>"       ≡ "<X> PT"  (word order is irrelevant)
* Slip filename hints (e.g. "Slip Gaji Alsut …") often name the clinic.
* Smaller |amount_diff_pct| wins ties.
* The credit's keterangan often contains a `FEE DOKTER` / `FEE DRG` / \
`HONOR` / `HONORARIUM` token plus a corporate sender — match the sender \
against the slip's institution_name and filename.

## Output

The user sends two JSON arrays: `slips` and `credits`. Each item has an \
`id`. Return strict JSON: \
`{{"pairs": [{{"slip_id": int, "credit_id": int|null, "confidence": 0..1, \
"reason": "≤30 words"}}]}}`. Include EVERY slip in the output (use \
credit_id=null for slips with no acceptable match). Reason must name the \
rule and signal that drove your choice."""


def _response_schema() -> dict:
    return {
        "name": "match_pairings",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "pairs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "slip_id": {"type": "integer"},
                            "credit_id": {"type": ["integer", "null"]},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "reason": {"type": "string"},
                        },
                        "required": ["slip_id", "credit_id", "confidence", "reason"],
                    },
                }
            },
            "required": ["pairs"],
        },
        "strict": True,
    }


# ----------------------------- public entry --------------------------------

async def match_month(
    month: str,
    slips: list[ParsedSlip],
    credits: list[GajiCredit],
) -> tuple[list[MatchPair], list[ParsedSlip], list[GajiCredit], Optional[str]]:
    """Pair this month's slips with this month's Gaji credits.

    Returns:
        (matches, unmatched_slips, unmatched_credits, error_message_or_None).

    The function is robust: on any LLM error every slip + credit returns as
    unmatched and the error message is bubbled up for the audit field.
    """
    if not slips or not credits:
        # Nothing to match — degenerate but valid case.
        return [], list(slips), list(credits), None

    settings = get_settings()
    tolerance = settings.match_amount_tolerance_pct

    slip_payload = [
        {
            "id": i,
            "source_file": s.source_file,
            "worker_name": s.worker_name,
            "institution_name": s.institution_name,
            "total_paid": s.total_paid,
            "month": s.month,
        }
        for i, s in enumerate(slips)
    ]
    credit_payload = [
        {
            "id": i,
            "source_file": c.source_file,
            "tanggal": c.tanggal,
            "keterangan": c.keterangan,
            "amount": c.amount,
            "month": c.month,
        }
        for i, c in enumerate(credits)
    ]

    client = AsyncAzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        timeout=settings.llm_request_timeout_s,
    )

    try:
        completion = await client.chat.completions.create(
            model=settings.azure_openai_deployment,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(tolerance=tolerance)},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"month": month, "slips": slip_payload, "credits": credit_payload},
                        ensure_ascii=False,
                    ),
                },
            ],
            response_format={"type": "json_schema", "json_schema": _response_schema()},
            temperature=0,
        )
        raw = completion.choices[0].message.content or "{}"
        decoded = json.loads(raw)
        pairs = decoded.get("pairs", [])
    except (APIError, APITimeoutError, json.JSONDecodeError, KeyError) as exc:
        logger.warning("matcher LLM call failed for month %s: %s", month, exc)
        return [], list(slips), list(credits), f"{month}: {exc}"

    # Build response, enforcing hard rules in code (defense-in-depth — the LLM
    # has been observed proposing pairs that violate rule 2 with an explicit
    # admission, so we don't rely on it for safety).
    used_credit_ids: set[int] = set()
    matched_slip_ids: set[int] = set()
    matches: list[MatchPair] = []
    for p in pairs:
        sid = p.get("slip_id")
        cid = p.get("credit_id")
        if not isinstance(sid, int) or sid < 0 or sid >= len(slips):
            continue
        if cid is None:
            continue  # the LLM said "no match" — fine, slip will surface as unmatched
        if not isinstance(cid, int) or cid < 0 or cid >= len(credits):
            continue

        slip = slips[sid]
        credit = credits[cid]
        if not slip.total_paid:
            logger.info("matcher: skipping slip %d (month %s) — total_paid is missing", sid, month)
            continue

        # Rule 3: each credit assigned to at most one slip.
        if cid in used_credit_ids:
            logger.info("matcher rule-3 violation rejected: credit %d reused for slip %d (month %s)",
                        cid, sid, month)
            continue

        # Rule 2: amount tolerance (the spec's hard rule).
        diff_rp = float(credit.amount) - float(slip.total_paid)
        diff_pct = diff_rp / float(slip.total_paid)
        if abs(diff_pct) > tolerance:
            logger.info("matcher rule-2 violation rejected: slip %d ↔ credit %d in %s "
                        "has |diff_pct|=%.3f > tolerance=%.3f",
                        sid, cid, month, abs(diff_pct), tolerance)
            continue

        # Accepted.
        used_credit_ids.add(cid)
        matched_slip_ids.add(sid)
        try:
            credit_day = int(credit.tanggal.split("-")[2])
        except (IndexError, ValueError):
            credit_day = 0
        matches.append(MatchPair(
            slip=slip,
            credit=credit,
            confidence=float(p.get("confidence") or 0.0),
            reason=str(p.get("reason") or ""),
            amount_diff_rp=diff_rp,
            amount_diff_pct=diff_pct,
            days_off=0 if credit_day == 0 else abs(credit_day - 28),
        ))

    unmatched_slips = [s for i, s in enumerate(slips) if i not in matched_slip_ids]
    unmatched_credits = [c for i, c in enumerate(credits) if i not in used_credit_ids]
    return matches, unmatched_slips, unmatched_credits, None


async def match_all_months(
    by_month: dict[str, tuple[list[ParsedSlip], list[GajiCredit]]],
) -> tuple[list[MatchPair], list[ParsedSlip], list[GajiCredit], list[str]]:
    """Fan-out: one concurrent LLM call per month.

    Returns:
        (all_matches, all_unmatched_slips, all_unmatched_credits, errors).
    """
    coroutines = [match_month(m, slips, credits) for m, (slips, credits) in by_month.items()]
    results = await asyncio.gather(*coroutines, return_exceptions=False)

    all_matches: list[MatchPair] = []
    all_unmatched_slips: list[ParsedSlip] = []
    all_unmatched_credits: list[GajiCredit] = []
    errors: list[str] = []
    for matches, unmatched_slips, unmatched_credits, err in results:
        all_matches.extend(matches)
        all_unmatched_slips.extend(unmatched_slips)
        all_unmatched_credits.extend(unmatched_credits)
        if err:
            errors.append(err)
    return all_matches, all_unmatched_slips, all_unmatched_credits, errors
