# `ocr_match` — salary slip ↔ bank mutation matcher

**Status:** v0.2 implemented (deterministic matcher — see §15 Change Log)
**Original date:** 2026-06-02
**Last updated:** 2026-06-03
**Author:** Project owner
**Companion services:** [`ocr_slip`](../../../ocr_slip/) (salary-slip parser), [`ocr_mutasi`](../../architecture.md) (bank-statement parser + LLM classifier)

This spec describes a standalone third service that pairs **salary slips** (parsed by `ocr_slip`) with **Gaji-classified bank-credit rows** (parsed and classified by `ocr_mutasi`). It is the third element in a three-service toolkit and is intentionally a thin orchestrator — it never re-parses PDFs itself.

---

## 1. Goals & Non-Goals

### Goals
- Given **N salary-slip PDFs** and **M bank-statement PDFs** for the same person, produce a structured list of slip → bank-credit pairs.
- Handle fuzzy clinic-name mapping that pure heuristics can't (e.g. slip filename `Slip Gaji Alsut …` ↔ bank credit `FEE DOKTER | KLINIK CONTOH PT`).
- Surface **unmatched slips** (employer says they paid, no bank record found) and **unmatched Gaji credits** (bank shows income, no slip uploaded) so the user can see both gaps.
- Reuse the existing services unchanged: `ocr_match` calls `ocr_slip` and `ocr_mutasi` over HTTP, never reaches into their internals.
- Same operational shape as the other two services: FastAPI app, isolated venv, an `/upload` HTML page for browser testing.

### Non-Goals (v1)
- Building a new bank parser or slip parser. Those live in their own repos.
- Persisting matches. The response is returned to the caller; nothing is stored.
- Multi-account matching (slips for person A reconciled against bank for person B). Single-account only.
- Authentication / rate limiting. Lives behind the same internal gateway as the other two services.

---

## 2. Architecture

### 2.1 High-level flow

```
                Browser / client
                      │
        ┌─────────────┴─────────────┐
        │ POST /api/v1/match        │
        │ multipart/form-data:      │
        │   slips      = N PDFs     │
        │   mutations  = M PDFs     │
        └─────────────┬─────────────┘
                      ▼
       ┌──────────────────────────────┐
       │   ocr_match   (port 8200)    │
       │                              │
       │   1. fan-out parsing         │
       │   2. per-month LLM matcher   │
       │   3. assemble response       │
       └──────┬───────────┬───────────┘
              │           │
   HTTP POST  │           │  HTTP POST
              ▼           ▼
    ┌─────────────┐  ┌─────────────────────────────┐
    │ ocr_slip    │  │ ocr_mutasi                  │
    │ :8100       │  │ :8000                       │
    │ POST /parse │  │ POST /…/extract-batch       │
    │             │  │ (classifies credits as Gaji)│
    └─────────────┘  └─────────────────────────────┘
```

The matcher **never re-parses PDFs itself.** Every parser lives in exactly one place. If `ocr_slip` or `ocr_mutasi` later gain support for a new format, `ocr_match` automatically benefits without code changes.

### 2.2 Why a separate service (not an endpoint in an existing one)?

- **`ocr_slip` is in a separate git repo.** Adding a new endpoint there would split feature ownership across two repositories and require its venv to also carry the Azure OpenAI SDK that `ocr_match` needs.
- **`ocr_mutasi` already has 5 routes** with a clear classification mission. Adding cross-service orchestration would dilute its focus.
- **A 3rd standalone service mirrors the user's existing operational pattern** (8000 + 8100; 8200 fits naturally as the next port).
- The orchestration code is small (< 300 lines). It does not justify reorganising the existing services.

---

## 3. Components

Module-per-responsibility. Every unit is small enough to hold in one mental frame.

| Module | Responsibility |
|---|---|
| `ocr_match/config.py` | Load `.env` via `pydantic-settings`: `AZURE_OPENAI_*`, `OCR_SLIP_URL`, `OCR_MUTASI_URL`, `MATCH_AMOUNT_TOLERANCE_PCT`, `LLM_REQUEST_TIMEOUT_S`. |
| `ocr_match/models.py` | Pydantic types: `ParsedSlip` (re-uses ocr_slip's `ParsedDocument` shape), `GajiCredit` (re-uses ocr_mutasi's `BatchClassifiedCredit` shape), `MatchPair`, `MatchAudit`, `MatchResponse`. |
| `ocr_match/upstream.py` | Two thin typed HTTP clients: `parse_slips(pdfs) → list[ParsedSlip]` and `extract_mutations(pdfs) → list[GajiCredit]`. Each is one function. Raises `UpstreamUnreachableError` on connection failure. |
| `ocr_match/matcher.py` | The LLM call. Groups slips + credits by month, runs one Azure OpenAI call per month with the prompt in §6, returns `list[MatchPair]`. Uses structured-output JSON schema. |
| `ocr_match/pipeline.py` | Orchestrator. Single function `run(slip_pdfs, mutation_pdfs) → MatchResponse`. Steps: fan-out parse → filter `type=="CR" and category=="Gaji"` → group by month → matcher → assemble. |
| `ocr_match/api.py` | FastAPI app, `/api/v1/match`, `/upload` HTML, `/`, `/health`, `/favicon.ico`. ~150 lines. Same OpenAPI-3.0.3 patch as the other two services so Swagger UI renders file pickers correctly. |

---

## 4. Data model

All public response types are Pydantic so they double as FastAPI response schemas.

```python
class ParsedSlip(BaseModel):
    source_file: str                # e.g. "Slip Gaji Alsut drg. <NAME> - Feb 2025.pdf"
    worker_name: str | None
    institution_name: str | None
    total_paid: float | None
    pokok: float
    tax: float
    incentive: float
    deduction: float
    other_deduction: float
    month: str                      # YYYY-MM, parsed from filename/text

class GajiCredit(BaseModel):
    # Mirrors ocr_mutasi.models.BatchClassifiedCredit, narrowed to Gaji rows
    source_file: str
    tanggal: str                    # ISO date
    keterangan: str
    amount: float
    saldo: float | None
    page: int
    confidence: float | None        # the ocr_mutasi classifier's own confidence
    reason: str | None
    month: str                      # YYYY-MM, derived from tanggal

class MatchPair(BaseModel):
    slip: ParsedSlip
    credit: GajiCredit
    confidence: float               # match confidence, 0..1 (from the matcher LLM)
    reason: str                     # ≤ 30 words explaining the pairing
    amount_diff_rp: float           # signed: credit.amount - slip.total_paid
    amount_diff_pct: float          # signed: amount_diff_rp / slip.total_paid
    days_off: int                   # how far credit.tanggal is from the slip's expected month-end

class MatchAudit(BaseModel):
    slip_count: int
    credit_count: int               # how many Gaji credits were considered
    matched_count: int
    months_processed: list[str]     # ["2025-02", "2025-03", "2025-04"]
    matcher_errors: list[str]       # LLM failures, one per failed month
    upstream_errors: list[str]      # ocr_slip / ocr_mutasi failures

class MatchResponse(BaseModel):
    matches: list[MatchPair]
    unmatched_slips: list[ParsedSlip]
    unmatched_credits: list[GajiCredit]
    audit: MatchAudit
```

---

## 5. Response shape (example)

```jsonc
{
  "matches": [
    {
      "slip": {
        "source_file": "Slip Gaji Alsut drg. … - Feb 2025.pdf",
        "worker_name": "drg. <NAME>",
        "institution_name": "<CLINIC PT>",
        "total_paid": 5967000,
        "pokok": 4000000, "tax": 100000, "incentive": 2167000,
        "deduction": 100000, "other_deduction": 0,
        "month": "2025-02"
      },
      "credit": {
        "source_file": "8831502401_FEB_2025.pdf",
        "tanggal": "2025-02-05",
        "keterangan": "TRSF E-BANKING CR <ref> | FEE DOKTER | <CLINIC PT>",
        "amount": 5967000, "saldo": null,
        "page": 4, "confidence": 0.95,
        "reason": "FEE DOKTER + corporate sender → Gaji per rule 4",
        "month": "2025-02"
      },
      "confidence": 0.95,
      "reason": "Same month, exact amount, slip filename 'Alsut' maps to bank-credit sender 'ALAM SUTERA'",
      "amount_diff_rp": 0,
      "amount_diff_pct": 0.0,
      "days_off": 0
    }
    // … other matches
  ],
  "unmatched_slips": [],
  "unmatched_credits": [],
  "audit": {
    "slip_count": 6,
    "credit_count": 6,
    "matched_count": 6,
    "months_processed": ["2025-02", "2025-03", "2025-04"],
    "matcher_errors": [],
    "upstream_errors": []
  }
}
```

---

## 6. Matching algorithm

> **v0.2 amendment:** what's described in this section was the original LLM-based design. After testing against real data (`slip_david/`) we discovered that genuine pairs agree to the **rupiah** (Rp 0 diff) and follow a deterministic **X+1 payroll-lag** pattern (slip for month X → bank credit in month X+1, sometimes X). The implementation switched to a deterministic exact-match algorithm; the LLM is no longer in the matching path. See §15 Change Log for the rationale and §11 for the updated validation. The text below is retained for design-history continuity.

### 6.1 Pipeline

```
1.  Concurrent fan-out:
      ─ ocr_slip:8100/parse                    (with the N slip PDFs)
      ─ ocr_mutasi:8000/…/extract-batch        (with the M bank PDFs, classify=true)
2.  From ocr_mutasi response, keep credits where category == "Gaji".
3.  Tag both lists with `month` (YYYY-MM):
      ─ slips: parse from filename (matches `\b(Jan|Feb|...|Dec)\s+(\d{4})\b`)
               fallback: parse slip text from worker_name section
      ─ credits: derive from `tanggal[:7]`
4.  Group by month → { "2025-02": (slips_feb, credits_feb), … }
5.  For each month bucket, one Azure OpenAI call with the prompt in §6.2.
6.  Assemble MatchResponse:
      ─ matches: every (slip, credit) the LLM paired
      ─ unmatched_slips: slips returned with credit_id = null
      ─ unmatched_credits: Gaji credits the LLM didn't assign to any slip
```

### 6.2 LLM prompt

System prompt structure (same shape as the classifier — explicit rules):

```
You pair Indonesian salary slips with bank-credit rows that paid them.

You receive ONE month's worth at a time. Output JSON pairing each slip with
either ONE credit or null.

Hard rules (a pairing is invalid if any fails):
  1. The credit's month MUST equal the slip's month.
  2. |credit.amount − slip.total_paid| / slip.total_paid ≤ 0.15
     (15% tolerance — covers small tax/fee differences).
  3. Each credit may be assigned to at most ONE slip.

Soft signals (use to disambiguate when multiple credits pass the hard rules):
  • Institution-name fuzzy match. Common Indonesian abbreviations:
      - "Alsut"   ≡ "Alam Sutera"
      - "Bintaro" → often "BSD" (Bumi Serpong Damai, neighbouring area)
      - "Jaktim"  ≡ "Jakarta Timur"
      - "PT <X>"  ≡ "<X> PT"  (word order)
  • Slip filename hints (e.g. "Slip Gaji Alsut …")
  • Smaller |amount_diff_pct| wins ties.

Return strict JSON: [{slip_id, credit_id|null, confidence, reason}]
Reason ≤ 30 words. Cite the rule and signal that drove the choice.
```

Output JSON schema enforced server-side via `response_format={"type":"json_schema","strict":true,...}`, mirroring the pattern in `ocr_mutasi.llm_classifier`.

### 6.3 Why per-month LLM calls

- Independence: a slip from Feb cannot match a credit from Mar, so cross-month context adds no signal.
- Bounded payload: each call covers ~2–4 slips + ~2–6 credits regardless of how many months were uploaded.
- Concurrency: month-N calls run in parallel via `asyncio.gather`, so a year of statements is bounded by the slowest single call (~2–3 s), not N × call latency.

### 6.4 Why 15% as the amount tolerance

Real slip-vs-bank discrepancies come from: tax withheld between gross/net (slip's `total_paid` may be net, bank credit may be gross or vice versa), transfer fees, weekend rounding, partial-month proration. 15% covers all of these without admitting wildly different amounts. The threshold is a config knob (`MATCH_AMOUNT_TOLERANCE_PCT`) so it can be tightened or loosened per deployment.

### 6.5 Sign convention for diff fields

`amount_diff_rp` and `amount_diff_pct` are **signed**, computed as `credit.amount − slip.total_paid`. A positive value means the bank credited more than the slip claimed; negative means less. The matcher does not interpret the sign — that's left to whoever consumes the response (e.g. a v2 tax-reconciliation feature).

---

## 7. API surface

| Method | Path | Schema? | Purpose |
|---|---|---|---|
| `POST` | `/api/v1/match` | ✓ | Two file groups (`slips`, `mutations`) → `MatchResponse` |
| `GET`  | `/health` | ✓ | `{"status":"ok","version":"…"}` |
| `GET`  | `/` | hidden | `307` redirect to `/upload` |
| `GET`  | `/upload` | hidden | Self-contained HTML page with **two** drop zones (slips, mutations) and one Match button |
| `GET`  | `/favicon.ico` | hidden | `204` |
| `GET`  | `/docs`, `/redoc`, `/openapi.json` | n/a | FastAPI defaults, with OpenAPI 3.0.3 + schema-patch for file pickers (same pattern as ocr_mutasi & ocr_slip) |

### 7.1 `/upload` page

Two drop zones stacked vertically:
1. **Salary slips** — accepts multiple PDFs
2. **Bank statements** — accepts multiple PDFs

After submission, the page renders **matched pairs** as cards (slip metadata | arrow | credit metadata, with confidence and amount diff prominent), then collapsible sections for **unmatched slips** and **unmatched credits**. Bottom: collapsible raw-JSON.

Same colour palette and HTML/CSS conventions as `ocr_mutasi`'s and `ocr_slip`'s upload pages so the three services feel like one toolkit.

---

## 8. Error handling

| Scenario | HTTP | Behaviour |
|---|---|---|
| Empty `slips` field | `400` | `{"detail":"Upload at least one salary slip PDF."}` |
| Empty `mutations` field | `400` | `{"detail":"Upload at least one bank-statement PDF."}` |
| `ocr_slip` unreachable | `503` | `{"detail":"ocr_slip not reachable at <url>"}`, no LLM call attempted |
| `ocr_mutasi` unreachable | `503` | Same shape |
| Upstream returns `≥ 400` | `502` | `{"detail":"Upstream <name> returned <status>: <body>"}` |
| LLM call fails for one month | `200` | That month's slips and credits return as unmatched; `audit.matcher_errors` records the failure |
| LLM returns malformed JSON | `200` | Same as above (schema-strict mode catches this, but defensive try/except still wraps it) |
| Zero Gaji credits found | `200` | All slips return as `unmatched_slips`; `unmatched_credits == []` |

Logging follows the ocr_mutasi pattern: client faults log as `WARNING` with one line; upstream failures log as `ERROR` with the upstream response body; unexpected exceptions log as `ERROR` with full traceback.

---

## 9. Configuration (env vars)

Loaded once at startup via `pydantic-settings`. `.env.example` is committed; `.env` is gitignored.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | yes | — | Azure OpenAI resource URL |
| `AZURE_OPENAI_API_KEY` | yes | — | Key |
| `AZURE_OPENAI_API_VERSION` | yes | `2025-01-01-preview` | API version |
| `AZURE_OPENAI_DEPLOYMENT` | yes | `gpt-4.1-mini` | Deployment name |
| `OCR_SLIP_URL` | no | `http://127.0.0.1:8100` | Base URL of the slip parser |
| `OCR_MUTASI_URL` | no | `http://127.0.0.1:8000` | Base URL of the mutation parser |
| `APP_HOST` | no | `0.0.0.0` | Bind address |
| `APP_PORT` | no | `8200` | Bind port |
| `LLM_REQUEST_TIMEOUT_S` | no | `60` | Per-month LLM call timeout |
| `UPSTREAM_TIMEOUT_S` | no | `120` | Upstream HTTP call timeout |
| `MATCH_AMOUNT_TOLERANCE_RP` | no | `1` | Absolute rupiah tolerance for the exact-match rule (v0.2) |
| `MAX_FILES` | no | `50` | Per-upload cap (across both file groups combined) |

---

## 10. Project layout

```
ocr_match/                       ← project root (its own folder under the umbrella repo)
├── ocr_match/                   ← Python package
│   ├── __init__.py              ← version
│   ├── config.py                ← .env loader (pydantic-settings)
│   ├── models.py                ← Pydantic response types
│   ├── upstream.py              ← typed HTTP clients for ocr_slip and ocr_mutasi
│   ├── matcher.py               ← LLM call + structured-output schema
│   ├── pipeline.py              ← orchestrator (single entry point)
│   └── api.py                   ← FastAPI app + /upload HTML page
├── .env.example
├── .gitignore                   ← excludes .env, .venv/, *.pdf (PII)
├── README.md                    ← install / run / API / curl examples / troubleshooting
└── requirements.txt             ← fastapi, uvicorn, httpx, openai, pydantic-settings, python-multipart
```

Total expected size: ~700 lines of Python across 6 modules + a self-contained HTML page in `api.py` (~250 lines including CSS and JS).

---

## 11. Testing strategy

### 11.1 Smoke test using the provided test data

```
Input:  6 slip PDFs (slip_david/) + 3 bank PDFs (mutasi_david/)
Run:    PUT both file groups against POST /api/v1/match
Assert:
  • audit.slip_count    == 6
  • audit.credit_count  ≥ 6      (Gaji credits found in the 3 statements)
  • audit.matched_count ≥ 5      (most slips paired)
  • audit.matcher_errors == []
  • For every pair: |amount_diff_pct| ≤ MATCH_AMOUNT_TOLERANCE_PCT
  • months_processed == ["2025-02", "2025-03", "2025-04"]
```

### 11.2 Unit tests

- `matcher.py`: hand-built synthetic month buckets with known correct answers (covers the soft-signal cases: Alsut→Alam Sutera, amount within tolerance, multiple slips competing).
- `upstream.py`: mock both upstream services with `httpx.MockTransport`; assert connection-error → `UpstreamUnreachableError`.
- `pipeline.py`: mock both `upstream.parse_slips` and `upstream.extract_mutations`; assert correct month-grouping and unmatched-slip / unmatched-credit handling.

### 11.3 What the smoke test does NOT assert

It does not assert which **specific** credit pairs with the Bintaro slip — that's a judgement call between two plausible candidates (PT KLINIK CONTOH JAKARTA vs KLINIK CONTOH BSD). The test asserts the matcher made *a* sensible choice with valid hard-rule compliance, not the specific choice. Whether the LLM lands on JAKARTA TIM or KASTARA BSD for "Bintaro" is documented as an Open Question for v2 — the user may want to add a hand-curated synonym table.

---

## 12. Alternatives considered

| Alternative | Rejected because |
|---|---|
| **Pure rule-based matcher (no LLM)** | Fails on fuzzy clinic-name semantics (Alsut↔Alam Sutera). Could be added later as a fast path, but v1 is LLM-only for simplicity. |
| **Single LLM call across all months** | Wastes context on cross-month combinations that are by construction invalid (rule 1). Per-month calls are smaller, faster, parallelisable. |
| **New endpoint in ocr_mutasi** | Couples two existing services to a third feature. Each parser owns its scope; the matcher is a new scope and deserves its own service. |
| **Standalone but JSON-in / JSON-out only (no PDF upload)** | Forces every caller to manually call all three services. PDF upload to one endpoint is the natural UX. |
| **Library, not a service** | Inconsistent with the rest of the toolkit, all of which are FastAPI services with `/upload` pages. |
| **Hungarian assignment / global optimiser** | More code, marginal benefit when month buckets are this small (≤ ~5 slips × ≤ ~6 credits). LLM with hard rules is sufficient. |

---

## 15. Change Log

| Version | Date | What changed |
|---|---|---|
| v0.1 | 2026-06-02 | Initial design: LLM-based per-month matcher with hard rules (same month, ±15% amount tolerance, one credit per slip). Implemented as `matcher.py:match_all_months` calling Azure OpenAI structured output per month bucket. Validation against `slip_david/` produced 3/6 matches with most pairs within 5–10% amount diff but one valid pair (Alsut Apr) at 27.5% diff — flagged for rejection by server-side tolerance enforcement. |
| **v0.2** | 2026-06-03 | **Switched to deterministic exact-amount matching after observing the real domain pattern.** Testing revealed that genuine slip↔credit pairs in the user's data agree to the rupiah (Rp 0 diff in 4 of 4 confirmed pairs) and follow a consistent **X+1 payroll-lag pattern** (slip for month X paid in bank month X+1; sometimes same month). The matcher was rewritten as a pure-rule algorithm: index credits by month, for each slip try month X+1 then X, pick the first unused credit whose amount equals slip.total_paid within `MATCH_AMOUNT_TOLERANCE_RP` (default Rp 1). The LLM call was removed entirely from the matching path (still imported for a possible future tie-break path when collisions arise). `MatchPair` gained a `match_pattern` field reporting whether the X+1 or same-month bucket fired. The `MATCH_AMOUNT_TOLERANCE_PCT` percentage knob was replaced with `MATCH_AMOUNT_TOLERANCE_RP` (absolute rupiah). Re-validation against the same `slip_david/` produced 4/4 confirmable matches at Rp 0 diff in ~14 s wall clock, vs the v0.1 LLM-based path's 3/6 in ~25 s. Two unmatched slips (Apr 2025) and four unmatched credits are correctly retained as legitimate signals — they pair with PDFs the user did not upload (May statement, Jan slips, BSD-clinic slips). |

## 16. Open questions / future work

- **Synonym table for known abbreviations.** Once we have user feedback on a real year of data, a hand-curated `{ "Alsut": ["Alam Sutera"], "Bintaro": ["BSD"], … }` map could let the rule-based fast path handle most cases and only fall back to LLM for genuinely novel pairs. Out of scope for v1.
- **Cross-bank matching.** If the user has slips that paid into BCA but bank statements from BRI, we'd need to expand the matcher's scope. v1 assumes one bank account at a time.
- **Tax/fee reconciliation.** Currently the matcher accepts a 15% tolerance; it does not explain *why* there's a diff. A v2 could surface "likely PPh 21 withholding of X%" by comparing slip's `tax` field to the diff.
- **Persisting matched/unmatched pairs.** For audit workflows. Out of scope while the toolkit stays stateless.
- **Confidence calibration.** With real-world feedback, the LLM's `confidence` values can be checked for calibration (does 0.9 mean ~90% correct?) and the prompt tightened if not.

---

## 14. Validation against the included sample data

| Slip | Expected pair (LLM call result) |
|---|---|
| `Slip Gaji Alsut drg. <NAME> - Feb 2025.pdf` | Feb-05 `FEE DOKTER | <CLINIC ALAM SUTERA PT>` |
| `Slip Gaji Alsut drg. <NAME> - Mar 2025.pdf` | Mar-05 `FEE DOKTER | <CLINIC ALAM SUTERA PT>` |
| `Slip Gaji Alsut drg. <NAME> - Apr 2025.pdf` | Apr-07 `FEE DOKTER | <CLINIC ALAM SUTERA PT>` |
| `Slip Gaji Bintaro drg. <NAME> - Feb/Mar/Apr 2025.pdf` | Either `PT <CLINIC> JAKARTA TIM` or `<CLINIC> BSD` (judgement call — see §11.3) |

Acceptance: matched_count ≥ 5 and all rule-1 / rule-2 constraints satisfied. The 6th match may surface as unmatched if amounts diverge — that's a legitimate signal, not a failure.
