# OCR Mutasi — Architecture

**Status:** v0.6 (current)
**Date:** 2026-06-01
**Audience:** Engineers extending the parser set, the LLM prompts, or the HTTP surface.
**Companion doc:** [`README.md`](../README.md) (install, run, API examples).

This document explains **why** the system is built the way it is. For *how* to use it, see the README.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Goals & Non-Goals](#2-goals--non-goals)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Request Sequences](#4-request-sequences)
5. [Components](#5-components)
6. [Data Model](#6-data-model)
7. [Response Schemas](#7-response-schemas)
8. [Per-Bank Layout Reference](#8-per-bank-layout-reference)
9. [Configuration](#9-configuration)
10. [Error Model](#10-error-model)
11. [Validation & Quality Signals](#11-validation--quality-signals)
12. [Performance Characteristics](#12-performance-characteristics)
13. [Real-World Validation](#13-real-world-validation)
14. [Security & PII](#14-security--pii)
15. [Testing Strategy](#15-testing-strategy)
16. [Alternatives Considered](#16-alternatives-considered)
17. [Extending the System](#17-extending-the-system)
18. [Open Questions / Future Work](#18-open-questions--future-work)
19. [Change Log](#19-change-log)

---

## 1. Overview

OCR Mutasi is a backend service that ingests Indonesian bank-statement ("mutasi") PDFs and produces structured JSON, with credit transactions semantically classified as **Gaji** (fixed monthly salary), **THR** (religious-holiday allowance), **Bonus** (annual), **Insentif** (performance), or **Lainnya** (other).

Despite the project name, **no image OCR is performed**. Indonesian bank statements from BCA and BRI ship as digital PDFs with a clean embedded text layer. Reading that layer with `pypdfium2` is faster, deterministic, and far more accurate than rasterizing and OCR-ing. The "OCR" in the name is historical; the system is best described as a *PDF text-layer extractor + geometric table reconstructor + LLM classifier*.

The service exposes two business endpoints plus a small set of routes that exist purely for usability:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/mutations/extract` | One PDF in, one structured response. Quick inspection, debugging, ad-hoc API integration. |
| `POST` | `/api/v1/mutations/extract-batch` | Many PDFs in, **one** cross-month LLM classification. The recurring-monthly-deposit pattern is the strongest Gaji signal and is invisible to single-PDF classification. |
| `GET`  | `/upload` | Plain-HTML upload page with a single multi-file input. Bypasses Swagger UI's array editor (which insists on one slot per file) and shows per-category accordion details after the response arrives. |
| `GET`  | `/` | `307` redirect to `/upload` so the bare URL lands somewhere useful. |
| `GET`  | `/favicon.ico` | `204` to swallow browser auto-requests without 404-noise in the access log. |
| `GET`  | `/health` | Liveness probe. |
| `GET`  | `/docs`, `/redoc`, `/openapi.json` | FastAPI's built-in API explorers and the (custom-patched, see §5.5) OpenAPI 3.0.3 schema. |

Three supported bank layouts:

- **BCA "Rekening Tahapan"** — 5-column table, `DB`/no-suffix marks debit/credit.
- **BRI "BritAma"** — 6-column bilingual table, **separate** Debet and Kredit columns with `0.00` placeholder in the unused one.
- **Mandiri "Tabungan Mandiri" e-Statement** — 5-column bilingual table, `+`/`-` prefix on the Nominal column signals credit/debit, and the number format is *inverted* from BCA/BRI (`.` thousands, `,` decimals).

Bank is auto-detected from page 1 text; no client-side flag needed.

---

## 2. Goals & Non-Goals

### Goals
- **Lossless extraction.** Every transaction row (date, description, amount, type, balance, source page) is captured.
- **Determinism below the LLM.** Identical PDF input always yields identical extracted JSON. Only the classification step is non-deterministic (and even there we set `temperature=0`).
- **Cross-month-aware classification.** A year of statements processed in a single LLM call so recurring patterns are visible.
- **Clean error model.** Client faults (bad PDF, unsupported bank) are `422` with a one-line message; genuine server bugs are `500` with full traceback.
- **Single-file bank extensibility.** Adding a third bank means one new module and two registration lines.
- **Reuse the existing Azure OpenAI deployment** (`gpt-4.1-mini`).

### Non-Goals (v1)
- Frontend / UI.
- Bank layouts beyond BCA Rekening Tahapan, BRI BritAma, and Mandiri Tabungan e-Statement (the pattern to add more is documented in §17).
- Scanned-image PDFs without a text layer.
- Authentication / rate limiting (this lives behind an internal gateway).
- Persistence (responses are returned to the caller; nothing is stored).

---

## 3. High-Level Architecture

```
                         ┌──────────────┐
   Browser ─── GET / ───►│  /          │── 307 ─►/upload
   Browser ─ /upload ───►│  /upload    │── HTML (multi-file form)
                         │              │
   Frontend ── POST ─────►/api/v1/mutations/extract        ── one PDF ─────┐
   Frontend ── POST ─────►/api/v1/mutations/extract-batch  ── many PDFs ─┐ │
                         └──────────────┘                                 │ │
                            FastAPI (api.py)                              │ │
                                                                          ▼ ▼
                                              ┌──────────────────────────────┐
                                              │   pipeline.run / run_batch   │
                                              │   (orchestrator + dispatch)  │
                                              └──────────────┬───────────────┘
                       ┌─────────────────────────────────────┼─────────────────────────┐
                       ▼                                     ▼                         ▼
            ┌─────────────────────┐   ┌──────────────────────────────────┐  ┌──────────────────────┐
            │  pdf_extractor.py   │   │      parsers/__init__.py         │  │  llm_classifier.py   │
            │  pypdfium2 wrapper  │   │   detect_bank() + get_parser()   │  │  Azure OpenAI        │
            │  → list[TextChunk]  │   │                                  │  │  (gpt-4.1-mini)      │
            │  raises             │   │   parsers/common.py              │  │                      │
            │  InvalidPdfError    │   │   parsers/bca.py   (BCA)         │  │  classify_credits    │
            └─────────────────────┘   │   parsers/bri.py   (BRI)         │  │  classify_credits_   │
                                      │   parsers/mandiri.py (Mandiri)   │  │    batch             │
                                      └──────────────────────────────────┘  └──────────────────────┘
                                                       │                              │
                                                       └──────────────┬───────────────┘
                                                                      ▼
                                                ┌────────────────────────────────────┐
                                                │  ExtractionResponse                │
                                                │  BatchExtractionResponse           │
                                                │  (pydantic models in models.py)    │
                                                └────────────────────────────────────┘
```

Every box is one Python module with a single responsibility. Modules talk to each other through a small set of typed structures in `models.py` and `parsers/common.py`. The HTTP layer is intentionally thin — all behaviour lives in the modules behind it.

---

## 4. Request Sequences

### 4.1 `POST /api/v1/mutations/extract` (single PDF)

```
Client → API: multipart/form-data { file: <pdf> }
API: validate content_type, size (≤ MAX_PDF_BYTES), non-empty
API → pipeline.run(pdf_bytes, classify=True)
  pipeline → pdf_extractor.extract_chunks(pdf_bytes)
    → list[TextChunk]                          (raises InvalidPdfError on garbage)
  pipeline → parsers.detect_bank(chunks)       (raises UnsupportedBankError if no match)
    → "BCA" | "BRI"
  pipeline → parsers.get_parser(bank)
  parser.parse_header(chunks)
    → AccountHeader
  parser.parse_transactions(chunks, header)
    → ParseResult { transactions, parse_warnings, balance_warnings }
  pipeline filters: credits_only = [t for t in tx if t.type == "CR"]
  pipeline → llm_classifier.classify_credits(credits_only)   (one PDF's worth)
    → list[ClassifiedCredit]                   (returns category=None on Azure failure)
  pipeline → ExtractionResponse {...}
API → 200 OK + JSON
```

### 4.2 `POST /api/v1/mutations/extract-batch` (many PDFs)

```
Client → API: multipart/form-data { files: [<pdf1>, <pdf2>, ...] }
API: validate each file (content_type, size, non-empty)
API → pipeline.run_batch([(name, bytes), ...], classify=True)
  for each file:
    pipeline → _extract_one(filename, bytes)   → FileExtraction
       (same flow as single endpoint up through parse_transactions)
  pipeline assembles credits_with_source =
       [(filename, tx) for fe in file_results for tx in fe.transactions if t.type == "CR"]
  pipeline → llm_classifier.classify_credits_batch(credits_with_source)
       ── ONE LLM call carrying every credit from every PDF ──
    → list[ClassifiedCredit]                   (model sees cross-month patterns)
  pipeline tags each result with source_file
  pipeline aggregates audit.category_totals
  pipeline → BatchExtractionResponse {...}
API → 200 OK + JSON
```

The pivotal difference is **one** LLM call across all months instead of N. That call costs roughly one round-trip's latency (~25–30 s for 12 months) and lets the model exploit recurrence — the strongest Gaji signal.

---

## 5. Components

### 5.1 `pdf_extractor.py` — PDF text-layer reader

**Responsibility:** open a PDF and return every text rect on every page as a `TextChunk`, with coordinates normalised to a top-left origin.

**Why pypdfium2:**
- It's a thin wrapper around PDFium, the same engine Chrome uses → industrial-grade text extraction with quirks already smoothed over.
- Ships prebuilt wheels per OS — **no system dependency** (unlike `pdfplumber` which needs Poppler or `pdf2image`).
- MIT-licensed, fast (PDFium is C++), good text-rect granularity.

**Implementation notes:**
- `PdfDocument(input)` accepts bytes, file paths, or file-like objects.
- Per page we call `get_textpage().count_rects()` and `get_rect(i)` to enumerate text runs (each rect is a contiguous span of text on a baseline).
- PDFium uses a **bottom-up Y** coordinate system (origin at bottom-left). We convert to top-down so every downstream module can reason about "top" vs "bottom" naturally: `y_top = page_height - rect_top`.
- Chunks are returned sorted by `(page, y0, x0)` — stable and intuitive.
- **Error handling:** any `pypdfium2.PdfiumError` (malformed/truncated PDF) is caught and re-raised as `InvalidPdfError`. This keeps pypdfium2 imports isolated to this one module — the API layer doesn't need to know about a third-party exception type.

### 5.2 `parsers/` — per-bank table reconstruction

#### 5.2.1 `parsers/__init__.py`
Two public functions:

```python
detect_bank(chunks: list[TextChunk]) -> "BCA" | "BRI" | "Mandiri" | "UNKNOWN"
get_parser(bank: str)                -> ModuleType   # the bca / bri / mandiri module
```

`detect_bank` joins page-1 chunk text and looks for distinguishing tokens:

| Bank | Detection token(s) |
|---|---|
| BCA | `REKENING TAHAPAN` |
| BRI | `LAPORAN TRANSAKSI FINANSIAL` or `BRITAMA` |
| Mandiri | `TABUNGAN MANDIRI` or `MENARA MANDIRI` |

This is intentionally cheap and tolerant; it returns `"UNKNOWN"` when none match, which the pipeline converts to `UnsupportedBankError` → `422`.

#### 5.2.2 `parsers/common.py`
Shared, bank-agnostic helpers:

- `extract_amount(text)` — robust to BCA's `,` thousands / `.` decimals and tolerant of trailing markers like `DB`/`CR`.
- `cluster_lines(chunks, tol_ratio=0.6)` — Y-clustering to build visual lines. Tolerance is `max(2.5, 0.6 × median_chunk_height)` — clipped below to avoid splitting a line by sub-pixel baseline jitter.
- `field_value(chunks, label)` — find the chunk immediately to the right of a label chunk on the same Y-line (used for `NO. REKENING : 1234567890` etc.).
- `year_from_periode(periode)` — first 4-digit token wins; 2-digit tokens like `26` become `2026`; falls back to current year.
- `join_cell(chunks)` — left-to-right concatenation of a column's chunks on one line.
- `Row { cells: dict[str, str], page: int, y: float }` — bank-agnostic row type. Each parser fills `cells` with its own column keys.
- `ParseResult { transactions, parse_warnings, balance_warnings }` — return value of `parse_transactions`.

#### 5.2.3 `parsers/bca.py`
Implements the BCA Rekening Tahapan layout (5 columns).

Public functions:
- `parse_header(chunks)` → `AccountHeader`
- `parse_transactions(chunks, header)` → `ParseResult`

Algorithm (per page):
1. **Detect the column header** — find chunks whose text matches `TANGGAL / KETERANGAN / CBG / MUTASI / SALDO` on the same baseline (±3 pt). Record their (x0, x1) spans as `_BcaLayout`.
2. **Compute column boundaries** from the layout (see §8.1 for calibration constants).
3. **Cluster body chunks into visual lines** via `cluster_lines`.
4. **Bucket each line's chunks into columns** by x-center → boundary lookup → `Row`.
5. **Group rows into transaction blocks** — a new block begins when a row has both a parseable `DD/MM` date in TANGGAL *and* a parseable amount in MUTASI. Following rows without that pair are description-continuation lines.
6. **Convert each block to a `Transaction`** — parse date with PERIODE year, parse amount, set type = `DB` if MUTASI ends with `DB` else `CR` (cross-check via `TRANSAKSI DEBIT` / `BI-FAST CR` keywords).

Per-page column layout is cached and reused on pages where header detection fails (rare for BCA, but defensive).

#### 5.2.4 `parsers/bri.py`
Implements the BRI BritAma layout (6 columns, bilingual headers, separate Debet/Kredit).

Key differences from BCA:
- Header anchors are matched against the **Indonesian** label line (`Tanggal Transaksi` / `Uraian Transaksi` / `Teller` / `Debet` / `Kredit` / `Saldo`). The English subheader (`Transaction Date` / ...) sits ~10 pt below — its y-range is included in the body-cutoff so it isn't accidentally clustered with the first transaction row.
- Date format is `DD/MM/YY HH:MM:SS` (regex `DATE_TIME_RE`). The 2-digit year is expanded as `2000 + yy`; the PERIODE year is only a fallback.
- Debet and Kredit are **separate columns**. The "other" column carries a `0.00` placeholder. `type = "CR"` if `Kredit > 0`, `"DB"` if `Debet > 0`. Both non-zero in the same row is rare and treated as a warning.
- A summary block at the bottom of the final page (`Saldo Awal / Total Transaksi Debet / Total Transaksi Kredit / Saldo Akhir`) is **explicitly trimmed before parsing** via `_trim_at_summary` so its numbers don't pollute the last transaction.
- BRI's `Teller` column carries a User-ID; we surface it in the `Transaction.cbg` field for the response (closest analogue to BCA's CBG).

#### 5.2.5 `parsers/mandiri.py`
Implements the Mandiri "Tabungan Mandiri" e-Statement layout (5 columns: `No / Tanggal-Date / Keterangan-Remarks / Nominal-Amount / Saldo-Balance`).

Key differences from both BCA and BRI:
- **Indonesian number format** — `.` thousands, `,` decimals (`5.000.000,00` = 5,000,000). Uses a Mandiri-specific `_parse_id_amount` helper instead of `common.extract_amount`.
- **Sign-based DB/CR** — the Nominal column carries a `+` or `-` prefix (e.g. `+5.000.000,00`, `-2.500,00`). No separate columns, no `DB`/`CR` suffix. `type = "CR"` when sign is `+` (or absent), `"DB"` when sign is `-`.
- **Date format** — `DD MMM YYYY` with English month abbreviations (`Apr`, `May`, `Jun` …). Indonesian abbreviations (`Mei`, `Agt`, `Okt`, `Des`) are also accepted.
- **Multi-line transactions** — each transaction occupies 3–5 visual lines packed close together: a Keterangan main label on top, the date below it, an anchor line carrying the No + Keterangan detail + Nominal + Saldo, then the time below, then optional description continuation. Transactions are separated by a vertical gap of 30+ pt. The block-grouper packs consecutive lines whose y-gap is below `TX_GAP_THRESHOLD_PT = 18.0` into one transaction.
- **Account holder name** in the header section may be split across two chunks (`BUDI` on one line, `SANTOSO` on the next). The `_mandiri_field` helper concatenates them by looking in the value column within a small y-window.
- **End-of-statement trim** — a `Disclaimer` block at the bottom of the final page (and an "ini adalah batas akhir transaksi anda" marker) is trimmed before parsing.

### 5.3 `llm_classifier.py` — Azure OpenAI classification

Two public functions:

```python
classify_credits(credits: list[Transaction])
    -> (list[ClassifiedCredit], error_or_None)

classify_credits_batch(credits_with_source: list[tuple[str, Transaction]])
    -> (list[ClassifiedCredit], error_or_None)
```

**Common design:**
- One LLM call with **all** input credits at once (single-PDF or batch). The model gets to see every row in context.
- `temperature=0` for maximum determinism.
- The OpenAI client's **structured output** is used (`response_format={"type": "json_schema", "json_schema": ...}`) so the response deserialises directly into our Pydantic models with zero string parsing.
- Per-row output: `category`, `confidence` (0–1), `reason` (≤25 words).
- **Failure mode:** any `APIError`, `APITimeoutError`, `JSONDecodeError`, or `KeyError` is caught; credits are returned with `category=None`, and `audit.classifier_errors` carries the error message. The request returns `200`. The frontend is expected to surface the error.

**Single-PDF prompt (`SYSTEM_PROMPT`):**
Defines the four categories with their typical Indonesian/English signals. Instructs the model to bias toward `Lainnya` when uncertain. Sees only one month's credits.

**Batch prompt (`BATCH_SYSTEM_PROMPT`):**
Adds the critical sentence: *"The single strongest signal for Gaji: same amount on roughly the same day-of-month across multiple months, from the same source/system. Recurrence beats keywords."* Each credit in the payload includes its `source_file`, so the model can see e.g. that `SAP-DD TRANSACTION` appears in every monthly file. This is what enables Gaji detection for descriptions that look unremarkable in isolation.

**Output schema (`_RESPONSE_SCHEMA`):**
Strict JSON schema — `{classifications: [{id, category, confidence, reason}]}` — with `category` constrained to the four enum values. The `strict: True` flag asks Azure OpenAI to enforce the schema server-side.

### 5.4 `pipeline.py` — orchestrator

Two functions: `run(pdf_bytes, *, classify=True)` and `run_batch(list[(name, bytes)], *, classify=True)`. Also defines `UnsupportedBankError`.

Both call `detect_bank → get_parser → parse_header → parse_transactions → (optionally) classify_credits[_batch]` and assemble the response. The batch path additionally:
- collects credits across files with their `source_file`
- calls `classify_credits_batch` once
- aggregates per-category `count` and `sum` into `audit.category_totals`

`_extract_one(filename, bytes)` is the per-file helper used inside `run_batch` — it runs through extraction + parsing but *not* classification (that's deferred until all files are extracted).

### 5.5 `api.py` — FastAPI surface

Two business endpoints plus a small set of usability routes. All non-business routes are marked `include_in_schema=False` so the OpenAPI spec only advertises the API.

| Method | Path | Schema? | Purpose |
|---|---|---|---|
| `POST` | `/api/v1/mutations/extract` | ✓ | Single-PDF extraction + classification |
| `POST` | `/api/v1/mutations/extract-batch` | ✓ | Multi-PDF extraction + cross-month classification |
| `GET`  | `/health` | ✓ | Liveness; returns `{"status":"ok","version":<v>}` |
| `GET`  | `/` | hidden | `307` redirect to `/upload` so the bare URL lands somewhere useful |
| `GET`  | `/upload` | hidden | Self-contained HTML upload page (described in §5.6) |
| `GET`  | `/favicon.ico` | hidden | `204` to silence browser auto-requests |

Both POST endpoints:
- Validate `content_type` (`application/pdf` or `application/octet-stream`, falling back to `.pdf` filename match), non-empty, and `≤ MAX_PDF_BYTES`.
- Dispatch to `pipeline.run` / `pipeline.run_batch`.
- Handle errors per §10.

**OpenAPI version & schema patch.** FastAPI defaults to OpenAPI 3.1, but Swagger UI's bundled renderer doesn't fully implement the 3.1 `contentMediaType` keyword for multi-file uploads — it falls back to a "plain string array" widget. We force OpenAPI 3.0.3 on the app (`app.openapi_version = "3.0.3"`) **and** override `app.openapi` with a `_custom_openapi` post-processor that rewrites any residual
```json
{ "type": "string", "contentMediaType": "application/octet-stream" }
```
into the 3.0.3-native
```json
{ "type": "string", "format": "binary" }
```
which Swagger UI does render as a file input. The patch is generic — it walks the whole schema, not just specific paths — so it applies to any future endpoint that accepts files.

The HTTP layer is ~170 lines total. All real behaviour lives in `pipeline.py` and below.

### 5.6 `/upload` — built-in multi-file HTML page

Self-contained HTML page served inline from `api.py` (no template engine, no static asset directory). Pure HTML/CSS/JS — no framework.

Why it exists: Swagger UI's array editor renders `array of binary` as one file picker *per array slot*, with an "Add string item" button to grow the array. No OpenAPI schema (3.0 or 3.1) can express "single field accepting multiple files" in a way that Swagger UI renders as `<input type="file" multiple>`. The upload page bypasses that limitation:

- One native `<input type="file" multiple accept="application/pdf">` — the OS file picker handles multi-selection via Cmd/Ctrl-click or Shift-click.
- Submits via `fetch` to `/api/v1/mutations/extract-batch` with the same field name (`files`) repeated per file in `FormData`.
- After the response arrives, renders an accordion (one `<details>` per category) showing every transaction's date, source file, amount, description, LLM confidence, and reason. Gaji / THR / Bonus / Insentif expand by default; Lainnya stays collapsed (usually noisy). Category names are colour-coded.
- The full raw JSON response is also available behind a collapsible "Show raw JSON" toggle.
- Long descriptions wrap; the table scrolls horizontally on narrow screens.
- All user-visible text is HTML-escaped via a small `esc()` helper.

This page is intentionally simple — it's a *test/inspection* surface for engineers, not a production UI. The real product frontend talks to `/api/v1/mutations/extract-batch` directly.

---

## 6. Data Model

All types are Pydantic v2 models in `ocr_mutasi/models.py`. They're used both internally and as FastAPI response schemas — one source of truth.

| Type | Fields | Used by |
|---|---|---|
| `TextChunk` | `text`, `x0`, `y0`, `x1`, `y1`, `page` | extractor → parser |
| `AccountHeader` | `bank`, `no_rekening`, `nama`, `periode`, `mata_uang` | response top-level |
| `Transaction` | `tanggal` (ISO), `keterangan`, `cbg`, `amount`, `type` (`DB`\|`CR`), `saldo?`, `page` | response `transactions[]` |
| `ClassifiedCredit` | `Transaction` + `category` (`Gaji`\|`THR`\|`Bonus`\|`Insentif`\|`Lainnya`\|`null`), `confidence?`, `reason?` | single-PDF `credits[]` |
| `Audit` | `pages_processed`, `rows_detected`, `credit_count`, `debit_count`, `balance_warnings[]`, `parse_warnings[]`, `classifier_errors[]` | response `audit` |
| `ExtractionResponse` | `account`, `transactions[]`, `credits[]`, `audit` | `/extract` response |
| `FileExtraction` | `filename`, `account`, `transactions[]`, `audit` | batch `files[]` element |
| `BatchClassifiedCredit` | `ClassifiedCredit` + `source_file` | batch `credits[]` element |
| `CategoryTotal` | `count`, `sum` | batch `audit.category_totals` value |
| `BatchAudit` | `files_processed`, `transactions_total`, `credits_total`, `classifier_errors[]`, `category_totals: dict[str, CategoryTotal]` | batch `audit` |
| `BatchExtractionResponse` | `files[]`, `credits[]`, `audit` | `/extract-batch` response |

Per-parser internal types (`_BcaLayout`, `_BriLayout`, `Row`) live in the parser modules; they aren't part of the public schema.

---

## 7. Response Schemas

### 7.1 `/extract` (`ExtractionResponse`)

```jsonc
{
  "account": {
    "bank": "BCA",
    "no_rekening": "1234567890",
    "nama": "BUDI SANTOSO",
    "periode": "APRIL 2026",
    "mata_uang": "IDR"
  },
  "transactions": [
    {
      "tanggal": "2026-04-01",
      "keterangan": "TRANSAKSI DEBIT TGL: 01/04 | QR 008 | 00000.00DAMRI-0364",
      "cbg": null,
      "amount": 25000.0,
      "type": "DB",
      "saldo": 1475000.00,           // null when the statement omits it on that row
      "page": 1
    }
    // … every transaction in the PDF, in source order
  ],
  "credits": [
    {
      "tanggal": "2026-04-15",
      "keterangan": "BI-FAST CR BIF TRANSFER DR | 002 | BUDI SANTOSO",
      "cbg": null,
      "amount": 10000000.00,
      "type": "CR",
      "saldo": 11000000.00,
      "page": 4,
      "category": "Lainnya",          // Gaji | THR | Bonus | Insentif | Lainnya | null
      "confidence": 0.7,              // 0..1, or null if classification failed
      "reason": "No salary keywords; transfer from individual."
    }
  ],
  "audit": {
    "pages_processed": 11,
    "rows_detected": 134,
    "credit_count": 3,
    "debit_count": 131,
    "balance_warnings": [],           // running-balance mismatches
    "parse_warnings": [],             // rows that failed to parse
    "classifier_errors": []           // populated if Azure OpenAI failed
  }
}
```

### 7.2 `/extract-batch` (`BatchExtractionResponse`)

```jsonc
{
  "files": [
    {
      "filename": "Mutasi_Mei_2025.pdf",
      "account": { ...AccountHeader },
      "transactions": [ ...Transaction[] ],
      "audit": { ...Audit (per-file) }
    }
    // … one entry per uploaded file
  ],
  "credits": [
    {
      "source_file": "Mutasi_Mei_2025.pdf",
      "tanggal": "2025-05-23",
      "keterangan": "SAP-DD TRANSACTION",
      "amount": 9500000.0,
      "type": "CR",
      "saldo": 11041425.84,
      "page": 1,
      "category": "Gaji",
      "confidence": 0.95,
      "reason": "SAP-DD TRANSACTION recurs monthly with similar amount and timing, strong payroll signal."
    }
    // … all credit rows across all files, in upload order
  ],
  "audit": {
    "files_processed": 12,
    "transactions_total": 951,
    "credits_total": 98,
    "classifier_errors": [],
    "category_totals": {
      "Gaji":     { "count": 12, "sum": 120000000.00, "min": 9500000.00 },
      "THR":      { "count":  1, "sum":  23000000.00, "min": 23000000.00 },
      "Bonus":    { "count":  1, "sum":  37000000.00, "min": 37000000.00 },
      "Insentif": { "count":  1, "sum":   7000000.00, "min":  7000000.00 },
      "Lainnya":  { "count": 83, "sum":  72000000.00, "min":    10000.00 }
    }
  }
}
```

---

## 8. Per-Bank Layout Reference

### 8.1 BCA Rekening Tahapan

Observed coordinates on `contoh_mutasi.pdf` page 1 (in PDF points, top-down y):

| Column | Header x-span | Header center | Cell content x range (observed) |
|---|---|---|---|
| TANGGAL | 33.27 – 73.26 | 53.27 | dates at xc=53.88 (`01/04`) |
| KETERANGAN | 164.41 – 219.22 | 191.82 | main label xc=119 (`TRANSAKSI DEBIT`), detail xc=207–237 (`TGL: 01/04`, `QR 008`, `00000.00DAMRI-0364`) |
| CBG | 308.39 – 325.33 | 316.86 | usually empty |
| MUTASI | 380.81 – 410.98 | 395.90 | amount right-aligned xc≈417; `DB` suffix xc≈447 |
| SALDO | 500.86 – 528.46 | 514.66 | right-aligned xc≈546 |

**Column boundary formula** (`_column_boundaries` in `bca.py`):

```python
b1 = layout.tanggal.x1 + 8        # just past TANGGAL header right edge
b2 = layout.cbg.x0     - 3        # just before CBG header
b3 = layout.mutasi.x0  - 8        # left of MUTASI start (amounts overflow leftward)
b4 = layout.saldo.x0   - 10       # left of SALDO start (same reason)
```

**Why not "midpoint of adjacent header centers"?** Because the KETERANGAN header is *centered* text inside a wide cell. `(53 + 192) / 2 = 122.5` would assign the main-label content at xc=119 to TANGGAL — a bug we hit early on. The actual cell left edge is much closer to TANGGAL's right edge than to KETERANGAN's center, so anchoring boundaries on header *edges* (with small calibrated pads) is correct.

**Quirks:**
- `SALDO AWAL` is on the first transaction line but has no MUTASI → naturally rejected by the block-grouper (which requires both a date and an amount).
- The final-page summary (`SALDO AWAL : … MUTASI CR : … MUTASI DB : … SALDO AKHIR : …`) also has no dates and is rejected the same way; no explicit trim needed.
- `02/04 TRANSAKSI DEBIT / TANGGAL :01/04` — sometimes the value-date in the description column carries a colon variant; this lives in KETERANGAN and is preserved verbatim.

### 8.2 BRI BritAma

Observed coordinates on `mutasi_haswin/Mutasi_April_2026.pdf` page 1:

| Column | Header x-span (Indonesian) | Header center | Cell content x range |
|---|---|---|---|
| Tanggal Transaksi | 37.07 – 100.56 | 68.81 | date+time xc=69 (`01/04/26 08:51:27`) |
| Uraian Transaksi | 170.07 – 227.66 | 198.87 | left-aligned xc=110–290 (`Transfer BI-Fast ke Bank Lain - …`) |
| Teller | 305.41 – 324.98 | 315.20 | User-ID xc=313 (`0371863`) |
| Debet | 382.32 – 402.65 | 392.48 | right-aligned xc=417–434, may carry `0.00` placeholder |
| Kredit | 483.41 – 503.51 | 493.46 | right-aligned xc=515–529, may carry `0.00` placeholder |
| Saldo | 580.32 – 599.03 | 589.68 | right-aligned xc=610 |

**Column boundary formula** (`_column_boundaries` in `bri.py`):

```python
b1 = layout.tanggal.x1 + 5
b2 = layout.teller.x0  - 10
b3 = layout.debet.x0   - 12
b4 = (debet_center + kredit_center) / 2     # both right-aligned, equidistant
b5 = (kredit_center + saldo_center) / 2
```

**Quirks:**
- Two header lines: Indonesian (e.g. `Tanggal Transaksi`) above English (`Transaction Date`). The detector matches the Indonesian line and treats the next 12 pt of vertical space as still-header (so transaction rows below aren't clustered with the English subheader).
- Summary block at the bottom of the **last** page (Saldo Awal / Total Transaksi Debet / Total Transaksi Kredit / Saldo Akhir) is explicitly trimmed via `_trim_at_summary`. Without that, the totals row's `35,000,000.00 / 25,000,000.00 / 10,500,000.00 / 59,000,000.00` would get bucketed into URAIAN / TELLER / KREDIT / SALDO and could pollute the last transaction's block.
- Description-continuation lines (e.g. counter-party name `Budi Santoso` on the row below the date) are picked up as URAIAN-only rows and appended to the preceding transaction block.

### 8.3 Mandiri Tabungan e-Statement

Observed coordinates on `sample_mandiri.pdf` page 1:

| Column | Header x-span (Indonesian) | Header center | Cell content x range |
|---|---|---|---|
| No | 20.56 – 30.48 | 25.52 | sequence number `1`,`2`,… at xc≈22 |
| Tanggal | 52.12 – 81.72 | 66.92 | `01 Apr 2026` at xc≈73; time `08:56:51 WIB` on next line at xc≈75 |
| Keterangan | 124.56 – 167.72 | 146.14 | main label xc≈152 (`Transfer BI Fast`), detail xc≈138–207, long counter-party line xc≈207 |
| Nominal (IDR) | 380.64 – 431.52 | 406.08 | right-aligned `+5.000.000,00` xc≈402; `-2.500,00` xc≈412 |
| Saldo (IDR) | 529.24 – 570.52 | 549.88 | right-aligned `6.000.000,00` xc≈547 |

**Column boundary formula** (`_column_boundaries` in `mandiri.py`):

```python
b1 = layout.tanggal.x0   - 10        # No / Tanggal
b2 = layout.keterangan.x0 - 5        # Tanggal / Keterangan
b3 = layout.nominal.x0   - 30        # Keterangan / Nominal (Ket content can run far right)
b4 = (layout.nominal.x1 + layout.saldo.x0) / 2   # Nominal / Saldo
```

**Quirks:**
- **Bilingual headers** — Indonesian line (`Tanggal`, `Keterangan`, `Nominal (IDR)`, `Saldo (IDR)`) sits above the English subheader (`Date`, `Remarks`, `Amount (IDR)`, `Balance (IDR)`). The detector matches the Indonesian line; the next ~12 pt is treated as still-header so the first transaction row isn't clustered with the subheader.
- **Multi-line transactions** — each transaction occupies 3–5 visual lines (gaps of 4–6 pt) and is separated from the next by a 30+ pt gap. The block-grouper packs consecutive lines whose y-gap is below `TX_GAP_THRESHOLD_PT = 18.0` into one block.
- **Account name** — `Nama/` value can span two chunks (`BUDI` + `SANTOSO`); `_mandiri_field` collects every chunk in the value column within the label's y-window so the continuation is captured.
- **Disclaimer trim** — the footer ("Disclaimer" section and the "ini adalah batas akhir transaksi anda" marker) is trimmed before parsing so it can't be mistaken for a transaction.

**Validated against `sample_mandiri.pdf`:** 7 transactions extracted, sums match the document's own summary box exactly (Dana Masuk Rp 5,000,000.00, Dana Keluar Rp 674,000.00, Saldo Akhir Rp 5,326,000.00), 0 balance warnings, 0 parse warnings.

---

## 9. Configuration

All settings load from `.env` once at startup via `pydantic-settings` (singleton via `get_settings()` with `lru_cache`).

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | yes | — | Azure OpenAI resource URL |
| `AZURE_OPENAI_API_KEY` | yes | — | Key for the deployment |
| `AZURE_OPENAI_API_VERSION` | yes | `2025-01-01-preview` | API version |
| `AZURE_OPENAI_DEPLOYMENT` | yes | `gpt-4.1-mini` | Deployment name |
| `APP_HOST` | no | `0.0.0.0` | uvicorn bind address |
| `APP_PORT` | no | `8000` | uvicorn bind port |
| `LLM_REQUEST_TIMEOUT_S` | no | `30` | Per-request timeout (batch doubles this internally) |
| `MAX_PDF_BYTES` | no | `20_000_000` | Per-file upload cap (20 MB) |

`.env` is gitignored; `.env.example` is the committed template.

---

## 10. Error Model

The error model has two clean tiers, mapped through specific exception types so each layer only catches what it's responsible for.

### 10.1 Exception types

| Exception | Raised by | Meaning |
|---|---|---|
| `pdf_extractor.InvalidPdfError` | `extract_chunks()` | Wrapper around `pypdfium2.PdfiumError`. The bytes aren't a valid PDF (truncated, corrupt, not a PDF). |
| `pipeline.UnsupportedBankError` | `pipeline.run` / `run_batch` | `detect_bank` returned `"UNKNOWN"` — the PDF is valid but doesn't match any registered bank layout. |
| Any other `Exception` | anywhere | Unexpected — treated as a server bug. |

### 10.2 HTTP status mapping

| Status | When | Body |
|---|---|---|
| `200` | Success (even if Azure OpenAI failed — credits returned with `category: null` and `audit.classifier_errors` populated) | `ExtractionResponse` / `BatchExtractionResponse` |
| `400` | Upload missing/empty/wrong content-type | `{"detail": "..."}` |
| `413` | File exceeds `MAX_PDF_BYTES` | `{"detail": "<filename>: exceeds N bytes"}` |
| `422` | `InvalidPdfError`, `UnsupportedBankError`, or zero rows detected | `{"detail": "Could not read PDF: ..."}` |
| `500` | Any other exception (genuine server bug) | `{"detail": "Internal error while parsing PDF"}` |

### 10.3 Logging

| Severity | Trigger | Why |
|---|---|---|
| `INFO` | Every request (FastAPI/uvicorn access log) | Normal operation |
| `WARNING` | Client-fault rejection — `rejected upload (InvalidPdfError\|UnsupportedBankError): ...` | Visible but not noisy; no traceback because the cause is known |
| `ERROR` (with traceback) | Unexpected exception, via `logger.exception("unexpected pipeline failure")` | Genuine bug; ops needs the stack |

This split keeps log volume low while still surfacing real bugs immediately.

---

## 11. Validation & Quality Signals

The system emits several non-blocking signals so callers can judge result quality without parsing the transaction list themselves.

### 11.1 Balance continuity

`parse_transactions` maintains a `running_balance`. When the statement *prints* a `saldo` on a row:

```
expected = running_balance + (amount if CR else -amount)
if |expected - printed_saldo| > 0.5:
    audit.balance_warnings.append("p<page> <date>: running <expected> vs printed <printed>")
    running_balance = printed_saldo   # re-anchor so subsequent checks stay useful
```

`running_balance` is initialised at the first row that *has* a printed saldo (not the first row overall, since some pre-table rows like `SALDO AWAL` might lack one).

A non-empty `balance_warnings` is a strong signal that something is wrong (missed or duplicated transaction, wrong sign, layout drift). Real-world: 0 warnings across 951 transactions in 12 BRI months.

### 11.2 Parse warnings

Rows where a date parses but the amount doesn't (or vice versa) are added to `audit.parse_warnings` and **excluded from `transactions`** so partial/garbage rows don't enter the data. Real-world: 0 warnings across 951 transactions.

### 11.3 Classification confidence

Each `ClassifiedCredit.confidence` is the LLM's self-reported 0–1 score. Useful as a UI cue (e.g. surface anything < 0.5 for human review). Confidence ≠ correctness — it's the model's calibration — but in our validation runs the 12 monthly `SAP-DD` Gaji classifications all returned `0.95`, which matches their unambiguous recurrence.

### 11.4 Classifier errors

When the Azure call fails, every credit is returned with `category: null` and `audit.classifier_errors` carries a short string (e.g. `"classifier error: APITimeoutError"`). The HTTP response is still `200` so the caller gets the extraction data; the frontend should handle the absence of categories.

---

## 12. Performance Characteristics

Measured on a 2026-vintage Apple Silicon laptop, Python 3.12.4.

| Operation | Single-PDF | Batch (12 PDFs) |
|---|---|---|
| PDF rasterisation / text extraction | ~30 ms / PDF | ~360 ms total |
| Geometric table parsing | ~10 ms / PDF | ~120 ms total |
| LLM classification (network round-trip + inference) | ~1.5–2 s | ~25–28 s |
| **End-to-end** (with `classify=true`) | ~2.2 s | ~29 s |
| End-to-end with `classify=false` | ~60 ms | ~500 ms |

The LLM call dominates. Batch is *roughly the same wall time as one single call*, even with 12× the data — proof that a single LLM call across all months is far cheaper than 12 sequential ones, in addition to being more accurate.

Memory: peak ~80 MB for a 12-month batch (all PDFs fit comfortably).

---

## 13. Real-World Validation

### 13.1 BCA sample (`contoh_mutasi.pdf`)

| Metric | Result | Expected |
|---|---:|---:|
| Transactions parsed | 134 | 134 (per the document's own summary) |
| Debits | 131 | 131 |
| Credits | 3 | 3 |
| Sum of debits | Rp 19,000,000.00 | Rp 19,000,000.00 (exact) |
| Sum of credits | Rp 21,000,000.00 | Rp 21,000,000.00 (exact) |
| Balance warnings | 0 | 0 |
| Parse warnings | 0 | 0 |

### 13.2 BRI year (`mutasi_haswin/Mutasi_*.pdf` — 12 months)

| Metric | Result |
|---|---:|
| Files processed | 12 |
| Pages processed | 48 |
| Transactions total | 951 (853 DB + 98 CR) |
| Balance warnings (across the year) | 0 |
| Parse warnings (across the year) | 0 |
| Per-file DB/CR sums | match each PDF's own `Saldo Akhir = Saldo Awal − DB_total + CR_total` exactly |

### 13.3 Mandiri sample (`sample_mandiri.pdf`)

| Metric | Result | Expected |
|---|---:|---:|
| Bank auto-detection | `Mandiri` | `Mandiri` |
| Account name (multi-chunk reassembly) | `BUDI SANTOSO` | same |
| Transactions parsed | 7 | 7 (matches rendered statement) |
| Debits | 6 | 6 |
| Credits | 1 | 1 |
| Sum of debits (Dana Keluar) | Rp 674,000.00 | Rp 674,000.00 *(exact)* |
| Sum of credits (Dana Masuk) | Rp 5,000,000.00 | Rp 5,000,000.00 *(exact)* |
| Final saldo (Saldo Akhir) | Rp 5,326,000.00 | Rp 5,326,000.00 *(exact)* |
| Balance warnings | 0 | 0 |
| Parse warnings | 0 | 0 |

### 13.4 Cross-month classification (the prize)

Year-level totals returned by `/extract-batch`:

| Category | Count | Year sum (Rp) | Min single tx (Rp) | What it caught |
|---|---:|---:|---:|---|
| **Gaji**     | 12 | 120,000,000 |  9,500,000 | All 12 monthly `SAP-DD TRANSACTION` payroll deposits — recognised purely from cross-month recurrence (no salary keyword in descriptions) |
| **THR**      | 1  |  23,000,000 | 23,000,000 | 1× `THR_Islam_2026` (religious-holiday allowance) |
| **Bonus**    | 1  |  37,000,000 | 37,000,000 | 1× `BONUS_POOL_2025_1` (annual / structured) |
| **Insentif** | 1  |   7,000,000 |  7,000,000 | 1× `BONUS_INTERIM_2025` (performance-triggered, distinct from annual Bonus) |
| Lainnya      | 83 |  72,000,000 |     10,000 | P2P transfers, refunds, ECUTI leave allowances, etc. |

The same LLM call against per-month-isolated credits produced **zero** Gaji classifications. Cross-month context turned 0 → 12 with confidence 0.95 — the most concrete validation possible that the batch endpoint solves a real problem.

---

## 14. Security & PII

### 14.1 Data flow
- PDFs flow client → API → in-memory pipeline → response.
- Credit-row text (date, description, amount) is sent to Azure OpenAI for classification.
- Debits are **not** sent to the LLM (only credits).
- No transaction data is logged at INFO level (only filenames in WARNING messages on rejection).

### 14.2 Secrets
- `AZURE_OPENAI_API_KEY` lives in `.env` (gitignored).
- `.env.example` ships placeholders.
- No secrets are logged or surfaced in responses.

### 14.3 Inputs trusted
- `MAX_PDF_BYTES` caps memory pressure per upload (default 20 MB).
- `content_type` and filename extension are checked, but the real validation is `pypdfium2` (PDFium itself); we don't try to be smarter than the parser.
- No path traversal: filenames are only used as cosmetic labels in `BatchClassifiedCredit.source_file`. Nothing is written to disk.

### 14.4 Deployment posture
This service has no auth and no rate limiting. It's intended to live behind an internal gateway that handles both.

---

## 15. Testing Strategy

Current state: hand-validated against `contoh_mutasi.pdf` and 12 real BRI months (see §13). Automated tests are scaffolded but not yet wired into CI.

### Recommended test hierarchy

1. **Unit tests** (per parser, no PDF I/O): construct synthetic `TextChunk` lists and assert the parser produces expected `Transaction` rows. Cover:
   - Single-line transaction
   - Multi-line KETERANGAN / URAIAN
   - Missing SALDO (most rows)
   - Page-break header re-use (column layout cached from previous page)
   - Summary-block trim (BRI specifically)
2. **Integration test** (per parser): run the full `pipeline.run` on a sample PDF, assert counts and category totals.
3. **End-to-end HTTP test** (`fastapi.testclient.TestClient`):
   - `GET /health` → 200
   - `POST /extract` with valid PDF → 200, expected shape
   - `POST /extract` with garbage bytes → 422, error message
   - `POST /extract` with PDF lacking bank signature → 422 `UnsupportedBankError`
   - `POST /extract-batch` with multiple files → 200, expected `category_totals`
4. **LLM call is mocked in CI.** A separate `tests/manual/` script can exercise the real Azure call when needed.

---

## 16. Alternatives Considered

| Alternative | Why not |
|---|---|
| **PaddleOCR on rasterised pages** (v0.1 plan) | Inputs are always digital PDFs; OCR is slower, fuzzier, and adds a heavy dependency. |
| **`pdfplumber`** | Works, but slower on big PDFs and depends on `pdfminer.six` whose maintenance is uneven. `pypdfium2` ships precompiled wheels and uses the same engine as Chrome. |
| **A single unified parser** | We tried mid-process. BCA and BRI differ in column count, date format, debit/credit signalling, and even header language. Forcing a single parser would introduce per-bank conditionals in every step. A per-bank module with a tiny dispatcher is much cleaner. |
| **Regex-based classification (no LLM)** | The real BRI sample's salary deposits have no `GAJI`/`PAYROLL` keyword — they're `SAP-DD TRANSACTION`. Only an LLM with cross-month context catches this. |
| **One LLM call per credit row** | Slow (98× round trips for the BRI year), expensive (≈100× tokens of overhead), and pointless — the LLM gets *more* accurate when it sees more rows at once because it can spot patterns. |
| **Markdown intermediate instead of JSON** | Markdown loses types and forces the LLM to re-parse; JSON wins. (A markdown debug dump is still a useful side artifact for humans during development.) |
| **`PPStructure` table-recognition model** | Generic table-rec models mis-segment sparse financial tables with merged/multi-line cells and intermittent SALDO. Worth revisiting if we ever support unknown layouts at runtime. |

---

## 17. Extending the System

### 17.1 Adding a new bank

1. **Inspect the layout.** Extract chunks from a sample PDF and observe:
   - Header tokens (the strings to match in `detect_bank`).
   - Column header positions on page 1.
   - Cell content positions per column (left/right alignment, multi-line rules).
   - Date format and debit/credit signalling.
   - Any summary block at the end that needs trimming.
2. **Create `parsers/<bank>.py`.** Mirror the BCA or BRI module structure:
   - A `_<bank>Layout` dataclass with one (x0, x1) tuple per column.
   - `_detect_column_layout(page_chunks)` that returns the layout + header bottom-y.
   - `_column_boundaries(layout)` that derives x-thresholds. **Calibrate these against real chunks** — see §8 for how. Don't trust midpoint-between-header-centers.
   - `_chunks_to_rows`, `_group_into_blocks`, `_block_to_transaction` using the patterns from BCA/BRI.
   - Public `parse_header(chunks)` and `parse_transactions(chunks, header)`.
3. **Register in `parsers/__init__.py`:**
   ```python
   from . import bca, bri, <bank>
   
   def detect_bank(chunks):
       ...
       if "<unique token>" in page1_text:
           return "<BANK>"
   
   def get_parser(bank):
       ...
       if bank == "<BANK>": return <bank>
   ```
4. **Validate.** Run `pipeline.run(..., classify=False)` on a sample. Confirm:
   - `transactions` count matches the PDF's own summary, if any.
   - Sum(DB) and Sum(CR) match the PDF's totals.
   - `balance_warnings` is empty.

A single bank typically takes 100–300 lines of parser code plus 5 lines in `__init__.py`.

### 17.2 Adding a new category

1. Add the literal to `Category` in `models.py`.
2. Add the enum value to `_RESPONSE_SCHEMA` in `llm_classifier.py`.
3. Extend both `SYSTEM_PROMPT` and `BATCH_SYSTEM_PROMPT` to define the new category and its signals.
4. Extend `category_totals` initialisation in `pipeline.run_batch`.

### 17.3 Adding a new endpoint

The pattern is: validate inputs → call into `pipeline` → translate exceptions to status codes → return a Pydantic model. The HTTP layer should stay below ~150 lines total.

---

## 18. Open Questions / Future Work

- **Authentication & rate limiting** when this leaves the internal network.
- **Persistence / job queue** if classifying very long histories ever exceeds a single LLM request budget. (Today: 12 months = ~98 credits = ~25 s; we're nowhere near the ceiling.)
- **More banks** — Mandiri, BNI, CIMB. Each is one new `parsers/<bank>.py`.
- **Confidence-driven human review** — surface low-confidence (`< 0.5`?) classifications in the UI for user correction.
- **Multi-account merging** — if a user uploads statements from multiple of their own accounts, cross-account recurrence (same salary deposit across BCA and BRI months) would be an even stronger signal.
- **Scanned-PDF fallback** — bring PaddleOCR back as an opt-in path when `extract_chunks` returns near-zero text.
- **Streaming responses** for very large batches.

---

## 19. Change Log

| Version | Date | Highlights |
|---|---|---|
| v0.1 | (initial design) | PaddleOCR on rasterised BCA pages; single endpoint; credit-only filter. Never built. |
| v0.2 | 2026-05-31 | Switched to `pypdfium2`; built BCA parser, single `/extract` endpoint, Azure OpenAI classification; verified against `contoh_mutasi.pdf`. |
| v0.3 | 2026-05-31 | Added BRI BritAma parser, per-bank dispatch, parser subpackage; validated against 12 real BRI months. Surfaced cross-month classification gap. |
| v0.4 | 2026-05-31 | Added `/extract-batch` endpoint, `classify_credits_batch`, cross-month-aware prompt, year-level `category_totals` rollup. Resolved cross-month gap (0 → 12 Gaji detections). Refactored error model: `InvalidPdfError` wraps `pypdfium2.PdfiumError`; clean 422 vs 500 split with calibrated log severity. |
| v0.5 | 2026-06-01 | Several themes, broken out below. |
| **v0.6** | 2026-06-01 | **Breaking** — category schema expanded from 4 to 5 categories with clearer semantics: `Gaji` (fixed monthly salary), `THR` (Tunjangan Hari Raya — religious-holiday allowance), `Bonus` (annual / structured), `Insentif` (performance-triggered), `Lainnya` (other). The old `Tunjangan` catch-all is gone — its members redistribute into THR (religious-holiday) or Lainnya (generic monthly perks, ECUTI leave). `BONUS_INTERIM`-style performance pay now maps to `Insentif`; `BONUS_POOL`-style annual pay stays `Bonus`. `CategoryTotal` gains a `min` field (smallest single-tx amount per category; `null` when empty). The `/upload` page renders the min stat in each category's summary row and adds a cyan colour for the Insentif accordion. Both single-PDF and batch prompts rewritten to teach the LLM the new five-way distinction. |

**v0.5 in detail:**

- **Mandiri parser.** New `parsers/mandiri.py` for "Tabungan Mandiri" e-Statement. Handles the Indonesian number format (`.` thousands, `,` decimals — opposite of BCA/BRI), sign-prefixed Nominal column (`+`/`-` instead of a DB/CR suffix), multi-line per-transaction blocks grouped by y-gap, `DD MMM YYYY` dates with English month names, and account-name reassembly when the holder name spans two chunks. Validated end-to-end against `sample_mandiri.pdf` — every total matches the document's own summary.
- **Single-PDF prompt rewrite.** The previous prompt claimed monthly recurrence was the strongest Gaji signal — which the single endpoint can never observe. Rewrote `SYSTEM_PROMPT` to drop the unobservable hint and explicitly list `SAP-DD`, `ECUTI`, `BONUS_INTERIM`, and `BONUS_POOL` as payroll-system labels. Single-PDF Gaji recognition now works on real BRI data (was 0/1, now 1/1 in the test case).
- **`/upload` page.** New `GET /upload` route serves an inline HTML page with a single `<input type="file" multiple>` — the cleanest way around Swagger UI's per-array-item file-picker limitation. After upload, the page renders a per-category accordion (Gaji / THR / Bonus / Insentif / Lainnya) with every transaction's date, source file, amount, description, LLM confidence, and reason.
- **Swagger-friendly schema.** Pinned OpenAPI to 3.0.3 and added a `_custom_openapi` patch that rewrites `contentMediaType` (3.1) into `format: "binary"` (3.0.3-native) so Swagger UI's renderer actually shows a file picker for `/extract` and `/extract-batch`.
- **`/` redirect & favicon.** `GET /` now `307`s to `/upload` (was `/docs`). `GET /favicon.ico` returns `204` to keep the access log quiet.
- **BCA name extractor robustness.** `_first_name_line` no longer requires the address to begin with `TANAH` / `JL` / `KOTA` (a hardcode that only matched the sample). It anchors on the universal `KCP / KCU / KK / KANTOR` branch line and takes the chunk immediately below, with multi-chunk reassembly when the name wraps. Works across diverse BCA branches.
