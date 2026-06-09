# OCR Match

> Pair Indonesian **salary slips** with **bank-credit `Gaji` rows** from a bank statement. Third standalone service in the OCR toolkit; calls `ocr_slip` and `ocr_mutasi` upstream and runs a per-month LLM matcher.

| Service | Port | Role |
|---|---|---|
| `ocr_mutasi` | `8300` | Parses bank-statement PDFs, classifies credits into Gaji / THR / Bonus / Insentif / Lainnya |
| `ocr_slip`   | `8200` | Parses salary-slip PDFs into worker / institution / take-home / pokok / tax / incentive / deduction |
| **`ocr_match`** *(this service)* | `8400` | **Pairs the two** via an LLM matcher with hard rules + fuzzy company-name fallback |

The design spec is in [`docs/superpowers/specs/2026-06-02-ocr-match-design.md`](../docs/superpowers/specs/2026-06-02-ocr-match-design.md).

---

## Quick start

All four services live in one monorepo and share the **repo-root** `.venv`,
`requirements.txt`, and `.env`. Run everything **from the repo root** (not from
inside `ocr_match/`).

```bash
# from the ocr_mutasi repo root

# 1. venv + deps (once, shared by all services)
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. configure
cp .env.example .env
$EDITOR .env       # fill in AZURE_OPENAI_*; defaults for OCR_SLIP_URL / OCR_MUTASI_URL are fine

# 3. start the two upstream services + the matcher in separate shells
#    (all from the repo root):
#    Shell A:  .venv/bin/uvicorn ocr_mutasi.api:app --port 8300 --reload
#    Shell B:  .venv/bin/uvicorn ocr_slip.app:app   --port 8200 --reload
#    Shell C:  .venv/bin/uvicorn ocr_match.api:app  --port 8400 --reload

# 4. open the upload page
open http://127.0.0.1:8400/upload
```

You can also use curl. The endpoint takes two file groups (`slips` and `mutations`):

```bash
curl -X POST \
  $(for f in /path/to/slips/*.pdf;     do echo -n "-F slips=@$f "; done) \
  $(for f in /path/to/statements/*.pdf; do echo -n "-F mutations=@$f "; done) \
  http://127.0.0.1:8400/api/v1/match | jq .audit
```

---

## How matching works

```
POST /api/v1/match
   ├── slips      → forwarded to ocr_slip:/parse                → ParsedSlip[]
   ├── mutations  → forwarded to ocr_mutasi:/…/extract-batch    → BatchClassifiedCredit[]
   ↓                                                            (kept only where category == "Gaji")
Each item tagged with month = "YYYY-MM"
   ↓                                  (slip month from filename; credit month from tanggal)
Deterministic matcher (no LLM):
   for each slip with a known total_paid:
     try month X+1 first  (the common Indonesian payroll-lag pattern)
     fall back to month X (same-month payment)
       → first unused credit whose amount equals slip.total_paid
         (within MATCH_AMOUNT_TOLERANCE_RP, default Rp 1) wins
   ↓
Assemble MatchResponse: matches[] + unmatched_slips[] + unmatched_credits[] + audit
```

### Why the X+1 payroll-lag pattern matters

In Indonesian payroll convention, a **March slip is typically paid and shown in the bank statement in April**. Not always — sometimes pay date is in the same month. The matcher tries the `X+1` bucket first, then falls back to `X`. Each matched pair records which one fired in `match_pattern`:

| `match_pattern` | Meaning |
|---|---|
| `"next_month"` | Slip for month X, credit lands in month X+1 (the common case) |
| `"same_month"` | Slip for month X, credit lands in month X (fallback) |

The UI shows this as a small **`X+1`** or **`same month`** badge on each matched-pair card.

### Why no LLM

The user's actual payroll-vs-bank amounts agree **to the rupiah** for every genuine pair (`diff = Rp 0`, observed across all 4 real matches in the included sample). When the truth is that exact, a fuzzy matcher is the wrong tool — it admits noise, costs money, and adds latency. The matcher is now O(N+M) deterministic logic, runs in microseconds, and never needs an Azure round-trip.

The `LLM_REQUEST_TIMEOUT_S` config knob is kept for a possible future tie-break path (when ≥ 2 candidate credits have identical amounts and need fuzzy clinic-name disambiguation), but isn't called in v0.2.

---

## API

### `GET /health`
```json
{"status": "ok", "version": "0.1.0"}
```

### `POST /api/v1/match`

`multipart/form-data`. Two file groups, **both required**:

| field | type | description |
|---|---|---|
| `slips` | one or more PDFs | salary-slip PDFs |
| `mutations` | one or more PDFs | bank-statement PDFs |

**Status codes**
- `200` — success, returns `MatchResponse`
- `400` — empty / wrong content-type / one group empty
- `413` — total file count exceeds `MAX_FILES`
- `502` — an upstream service returned `≥ 400`
- `503` — an upstream service refused the connection
- `500` — unexpected server bug

**Response shape**

```jsonc
{
  "matches": [
    {
      "slip":   { /* full ParsedSlip from ocr_slip */ },
      "credit": { /* full GajiCredit from ocr_mutasi */ },
      "confidence": 1.0,                   // deterministic exact-match is always 1.0
      "reason": "Exact-amount match (diff Rp +0); slip month 2025-02 → credit month 2025-03 (X+1 payroll-lag pattern)",
      "amount_diff_rp": 0,                 // signed: credit.amount - slip.total_paid
      "amount_diff_pct": 0,
      "days_off": 5,                       // credit's day-of-month
      "match_pattern": "next_month"        // "next_month" (X+1) or "same_month" (X)
    }
  ],
  "unmatched_slips": [ /* slips with no acceptable credit */ ],
  "unmatched_credits": [ /* Gaji credits with no slip */ ],
  "audit": {
    "slip_count": 6,
    "credit_count": 8,
    "matched_count": 3,
    "months_processed": ["2025-02", "2025-03", "2025-04"],
    "matcher_errors": [],
    "upstream_errors": []
  }
}
```

### `GET /upload`

Self-contained HTML page with two drop-zones (Salary slips | Bank statements). After submission renders matched pairs as cards (slip ↔ credit with diff and reason), then unmatched lists, then a collapsible raw-JSON panel. Same visual language as the `/upload` pages on `ocr_slip` and `ocr_mutasi`.

### Other routes
- `GET /` — `307` redirect to `/upload`
- `GET /favicon.ico` — `204` (silences browser auto-requests)
- `GET /docs`, `/redoc`, `/openapi.json` — FastAPI defaults (file fields render as real pickers thanks to the OpenAPI 3.0.3 schema patch)

---

## Configuration

Loaded once at startup via `pydantic-settings`. Defaults in `config.py`; overrides via `.env`.

| Variable | Default | Purpose |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | — | Azure OpenAI resource URL *(required)* |
| `AZURE_OPENAI_API_KEY` | — | API key *(required)* |
| `AZURE_OPENAI_API_VERSION` | `2025-01-01-preview` | API version |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-4.1-mini` | Deployment name |
| `OCR_SLIP_URL` | `http://127.0.0.1:8200` | Base URL of the salary-slip parser |
| `OCR_MUTASI_URL` | `http://127.0.0.1:8300` | Base URL of the bank-statement parser |
| `APP_HOST` | `0.0.0.0` | Bind address |
| `APP_PORT` | `8400` | Bind port |
| `LLM_REQUEST_TIMEOUT_S` | `60` | Reserved for a future LLM tie-break path; not called in v0.2 |
| `UPSTREAM_TIMEOUT_S` | `120` | Upstream HTTP call timeout |
| `MATCH_AMOUNT_TOLERANCE_RP` | `1` | Absolute rupiah tolerance for the exact-match rule. Default is "essentially exact, with float-safety wiggle". Bump if your real-world data has cents-rounding or fee adjustments. |
| `MAX_FILES` | `50` | Per-upload cap across both groups combined |

---

## Project layout

A flat package at the repo root (same layout as `ocr_mutasi` / `ocr_classifier`).
Config files are centralized at the repo root — there is no per-service
`.env` / `requirements.txt` / `.venv`.

```
ocr_mutasi/                      ← repo root (shared .env, requirements.txt, .venv, .gitignore)
└── ocr_match/
    ├── __init__.py
    ├── config.py            ← reads the repo-root .env (pydantic-settings)
    ├── models.py            ← Pydantic response types
    ├── upstream.py          ← async HTTP clients for ocr_slip + ocr_mutasi
    ├── matcher.py           ← per-month Azure OpenAI call + rule enforcement
    ├── pipeline.py          ← orchestrator (single run() entry point)
    ├── api.py               ← FastAPI app + /upload HTML page
    └── README.md            ← this file
```

---

## Troubleshooting

| Symptom | HTTP | Likely cause | Fix |
|---|---|---|---|
| `ocr_slip not reachable at http://…:8200` | 503 | `ocr_slip` isn't running on the configured port | Start `ocr_slip` (see [`ocr_slip/README.md`](../ocr_slip/README.md)) or set `OCR_SLIP_URL` to where it actually runs |
| `ocr_mutasi not reachable …` | 503 | `ocr_mutasi` isn't running | Same — start it or update `OCR_MUTASI_URL` |
| `Upstream … returned 4xx/5xx: …` | 502 | Upstream got the request but rejected it (often: malformed PDF) | Check the `body` field in the error |
| `audit.matcher_errors` non-empty | 200 | The deterministic matcher cannot error; if this appears, it's an upstream-typed error misclassified — file a bug |
| All slips end up in `unmatched_slips` | 200 | The slip filename has no parseable month *or* there are 0 Gaji credits in the bank statement | Confirm filenames contain a month (`Feb 2025`, `April`, etc.); confirm the bank statement has classified Gaji credits via `ocr_mutasi`'s `/extract-batch` |
| A specific slip ends up unmatched but you can see what looks like the right credit | 200 | Amount differs by more than `MATCH_AMOUNT_TOLERANCE_RP` | Inspect both sides. If real-world payments legitimately differ by cents (PPh withholding rounding), bump `MATCH_AMOUNT_TOLERANCE_RP` to e.g. `100` or `1000`. Don't go higher than your domain expects — large tolerances admit cross-clinic collisions. |
| A slip for month X with no X+1 bank statement uploaded | 200 | The matcher tries `X+1` first then falls back to `X`. With only month-X uploaded and no X+1 credits, only the `same_month` pattern can fire — and only if your payroll happens to pay in the same month. | Upload the X+1 statement too, or accept that the slip is unmatched. |
| Swagger UI shows `Add string item` instead of file pickers | n/a | Stale build — the OpenAPI 3.0.3 patch wasn't applied | Restart uvicorn |

---

## Limitations (v1)

- Single-account matching only — slips for person A reconciled against bank for person A.
- Single bank per request — the bank-statement PDFs must all come from the same account.
- 15% tolerance is one number. Slip-vs-credit gaps from PPh-21 withholding can occasionally exceed this; widen the tolerance per deployment if your real-world distribution requires it.
- The LLM's choice between a `JAKARTA TIM` and a `BSD` credit for a slip labelled `Bintaro` is a judgement call, not a fact — see spec §13 for the eventual "hand-curated synonym table" plan.
- No persistence — the response is returned to the caller; nothing is stored.
- No auth, no rate limiting — runs behind an internal gateway.

---

## Real-world result on the included sample data

`slip_david/` (6 slips for a 2-clinic dentist) + `mutasi_david/` (3 monthly BCA statements):

| Metric | Result |
|---|---|
| Slips uploaded | 6 (Alsut / Bintaro × Feb / Mar / Apr 2025) |
| Gaji credits found | 8 across the 3 months |
| **Matched, exact-amount, X+1 pattern** | **4** (Feb-slip ↔ Mar-credit and Mar-slip ↔ Apr-credit, both clinics) |
| **Amount diff on each match** | **Rp 0** *(every pair, both clinics, both months)* |
| Unmatched slips | 2 (the **Apr 2025** slips, both clinics — would land in the May statement which wasn't uploaded) |
| Unmatched credits | 4 (Feb's two Alsut/Jakarta credits — paid from Jan slips not uploaded; 2 BSD credits — clinic not in the slip set) |
| Matcher errors | 0 |
| Wall clock | ~14 s end-to-end (mostly upstream PDF parsing; the matcher itself is microseconds) |

This is what *correct* looks like. The 2 unmatched slips and 4 unmatched credits are **legitimate signals**: Apr slips will pair with May credits (not uploaded), Feb credits paired with Jan slips (not uploaded), and the BSD clinic has no slip in this sample. No force-matches, no near-misses force-accepted, nothing pretended — the matcher honestly returns the exact set of pairs it can prove.
