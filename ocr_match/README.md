# OCR Match

> Pair Indonesian **salary slips** with **bank-credit `Gaji` rows** from a bank statement. Third standalone service in the OCR toolkit; calls `ocr_slip` and `ocr_mutasi` upstream and runs a per-month LLM matcher.

| Service | Port | Role |
|---|---|---|
| `ocr_mutasi` | `8000` | Parses bank-statement PDFs, classifies credits into Gaji / THR / Bonus / Insentif / Lainnya |
| `ocr_slip`   | `8100` | Parses salary-slip PDFs into worker / institution / take-home / pokok / tax / incentive / deduction |
| **`ocr_match`** *(this service)* | `8200` | **Pairs the two** via an LLM matcher with hard rules + fuzzy company-name fallback |

The design spec is in [`docs/superpowers/specs/2026-06-02-ocr-match-design.md`](../docs/superpowers/specs/2026-06-02-ocr-match-design.md).

---

## Quick start

```bash
# from ocr_mutasi project root
cd ocr_match

# 1. venv + deps
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. configure
cp .env.example .env
$EDITOR .env       # fill in AZURE_OPENAI_*; defaults for OCR_SLIP_URL / OCR_MUTASI_URL are fine

# 3. start the two upstream services in separate shells:
#    Shell A:  .venv/bin/uvicorn ocr_mutasi.api:app --port 8000 --reload     (from ocr_mutasi project root)
#    Shell B:  cd ocr_slip && PORT=8100 ./run_api.sh
#    Shell C:  .venv/bin/uvicorn ocr_match.api:app --port 8200 --reload

# 4. open the upload page
open http://127.0.0.1:8200/upload
```

You can also use curl. The endpoint takes two file groups (`slips` and `mutations`):

```bash
curl -X POST \
  $(for f in /path/to/slips/*.pdf;     do echo -n "-F slips=@$f "; done) \
  $(for f in /path/to/statements/*.pdf; do echo -n "-F mutations=@$f "; done) \
  http://127.0.0.1:8200/api/v1/match | jq .audit
```

---

## How matching works

```
POST /api/v1/match
   ├── slips      → forwarded to ocr_slip:/parse        → ParsedSlip[]
   ├── mutations  → forwarded to ocr_mutasi:/…/extract-batch → BatchClassifiedCredit[]
   ↓                                                    (kept only where category == "Gaji")
Each item is tagged with month = "YYYY-MM"
   ↓ (slip month from filename; credit month from tanggal)
Group by month → { "2025-02": (slips, gaji_credits), ... }
   ↓
For each month bucket  →  one Azure OpenAI call (gpt-4.1-mini)
   ↓                       (structured JSON output, temperature=0)
Server enforces hard rules before accepting a pair:
   ─ rule 1: months must match
   ─ rule 2: |credit − slip| / slip ≤ MATCH_AMOUNT_TOLERANCE_PCT  (default 15%)
   ─ rule 3: each credit assigned to at most ONE slip
   ↓
Assemble MatchResponse: matches[] + unmatched_slips[] + unmatched_credits[] + audit
```

### Decision rules in the LLM prompt

**Hard (server-side enforced):** same month • amount within ±15% • one-credit-per-slip.

**Soft (LLM-only — used to break ties between equally valid candidates):**
- `Alsut` ≡ `Alam Sutera`
- `Bintaro` → often `BSD` (Bumi Serpong Damai)
- `Jaktim` ≡ `Jakarta Timur`, `Jakbar` ≡ `Jakarta Barat`, …
- `PT <X>` ≡ `<X> PT` (word order irrelevant)
- Slip filename hints (`Slip Gaji Alsut …`)
- Smaller `|amount_diff_pct|` wins ties
- `FEE DOKTER` / `FEE DRG` / `HONOR` / `HONORARIUM` in the credit description + corporate sender → strong evidence

The LLM may propose hard-rule-violating pairs (it has been observed admitting "amount too high but no other credit fits better"). The matcher silently rejects those and the slip falls into `unmatched_slips`. A log line at INFO level records each rejection for audit.

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
      "confidence": 0.95,
      "reason": "Month match, amount within 15%, institution Alsut ≡ Alam Sutera",
      "amount_diff_rp": -366600,           // signed: credit.amount - slip.total_paid
      "amount_diff_pct": -0.058,
      "days_off": 23                       // |credit_day - 28|, rough month-end ref
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
| `OCR_SLIP_URL` | `http://127.0.0.1:8100` | Base URL of the salary-slip parser |
| `OCR_MUTASI_URL` | `http://127.0.0.1:8000` | Base URL of the bank-statement parser |
| `APP_HOST` | `0.0.0.0` | Bind address |
| `APP_PORT` | `8200` | Bind port |
| `LLM_REQUEST_TIMEOUT_S` | `60` | Per-month LLM call timeout |
| `UPSTREAM_TIMEOUT_S` | `120` | Upstream HTTP call timeout |
| `MATCH_AMOUNT_TOLERANCE_PCT` | `0.15` | Slip-vs-credit amount tolerance |
| `MAX_FILES` | `50` | Per-upload cap across both groups combined |

---

## Project layout

```
ocr_match/
├── ocr_match/
│   ├── __init__.py
│   ├── config.py            ← .env loader (pydantic-settings)
│   ├── models.py            ← Pydantic response types
│   ├── upstream.py          ← async HTTP clients for ocr_slip + ocr_mutasi
│   ├── matcher.py           ← per-month Azure OpenAI call + rule enforcement
│   ├── pipeline.py          ← orchestrator (single run() entry point)
│   └── api.py               ← FastAPI app + /upload HTML page
├── .env.example
├── .gitignore               ← excludes .env, .venv, *.pdf (PII)
├── requirements.txt
└── README.md                ← this file
```

---

## Troubleshooting

| Symptom | HTTP | Likely cause | Fix |
|---|---|---|---|
| `ocr_slip not reachable at http://…:8100` | 503 | `ocr_slip` isn't running on the configured port | Start `ocr_slip` (see [`ocr_slip/README.md`](../ocr_slip/README.md)) or set `OCR_SLIP_URL` to where it actually runs |
| `ocr_mutasi not reachable …` | 503 | `ocr_mutasi` isn't running | Same — start it or update `OCR_MUTASI_URL` |
| `Upstream … returned 4xx/5xx: …` | 502 | Upstream got the request but rejected it (often: malformed PDF) | Check the `body` field in the error |
| `audit.matcher_errors` non-empty | 200 | Azure OpenAI failed for that month | Slips and credits for that month all fall to unmatched; retry once Azure is healthy |
| All slips end up in `unmatched_slips` | 200 | The slip filename has no parseable month *or* there are 0 Gaji credits in the bank statement | Confirm filenames contain a month (`Feb 2025`, `April`, etc.); confirm the bank statement has classified Gaji credits via `ocr_mutasi`'s `/extract-batch` |
| A specific slip ends up unmatched but you can see the right credit | 200 | The amount diff exceeds `MATCH_AMOUNT_TOLERANCE_PCT`; check the `ocr_match` log for `rule-2 violation rejected` | Widen the tolerance via `MATCH_AMOUNT_TOLERANCE_PCT=0.25` (or whatever fits your actual diff distribution) |
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
| Slips uploaded | 6 (Alsut/Bintaro × Feb/Mar/Apr 2025) |
| Gaji credits found | 8 across the 3 months |
| Matched within ±15% | **3** (Alsut Feb, Bintaro Feb, Alsut Mar) |
| Unmatched slips | 3 (Bintaro Mar, Alsut Apr, Bintaro Apr — all exceed 15% diff) |
| Unmatched credits | 5 (3 to KLINIK CONTOH BSD, no slips for that clinic in the sample) |
| Matcher errors | 0 |
| Wall clock | ~25 s end-to-end (3 parallel LLM calls, one per month) |

That `matched_count = 3 of 6` is the *honest* answer for the test data — the Apr month's slip totals genuinely diverge from the bank credits by 27–37%, which exceeds the spec's 15% rule. Either the slip totals or the credit amounts are off (or both, due to gross/net interpretations). The pair is surfaced in `unmatched_slips`, not silently force-matched, so the user can investigate.
