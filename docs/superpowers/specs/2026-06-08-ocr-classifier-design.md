# `ocr_classifier` — document-type classifier

**Status:** Design approved — not yet implemented
**Date:** 2026-06-08
**Author:** Project owner
**Sibling services:** [`ocr_mutasi`](../../../ocr_mutasi/) (bank-statement parser + LLM classifier), [`ocr_slip`](../../../ocr_slip/) (salary-slip parser), [`ocr_match`](2026-06-02-ocr-match-design.md) (slip ↔ mutation matcher)

This spec describes a fourth standalone service that, given a document (PDF), runs it through the Google-Cloud PaddleOCR service, extracts the recognised text, and asks the LLM to classify the document as one of **`ktp` · `kk` · `sk` · `slip` · `mutasi` · `unknown`**. It is a pure classifier: it labels the document and stops. It does not parse the contents and does not route the document onward.

---

## 1. Goals & Non-Goals

### Goals
- Given one document PDF, return its type as one of `ktp`, `kk`, `sk`, `slip`, `mutasi`, or `unknown`, plus a coarse confidence and a one-line reason.
- Accept either a single document (`/classify`) or a batch (`/classify-batch`), classifying each file independently.
- Use the existing PaddleOCR service for text recognition and the existing Azure OpenAI deployment for the classification decision — no new ML infrastructure.
- Match the operational shape of `ocr_mutasi`: a flat package at the repo root, FastAPI app, JSON-schema-constrained LLM call, an `/upload` HTML page and Swagger UI for browser testing, sharing the repo-root `.env` / `requirements.txt` / `.venv`.

### Non-Goals
- Parsing or extracting structured fields from any document. That is the job of `ocr_mutasi` (mutasi) and `ocr_slip` (slip); KTP/KK/SK have no parser today and this service does not add one.
- Routing/forwarding the document to a downstream parser after classification. The caller decides what to do with the label.
- Persisting results, authentication, or rate limiting (lives behind the same internal gateway as the sibling services).
- Keyword/heuristic classification. The classification decision is made by the LLM, on the OCR text. (A deterministic keyword shortcut is explicitly out of scope per the project owner's instruction.)

---

## 2. Architecture

### 2.1 Layout (`ocr_mutasi` style — flat package at repo root)

The top-level folder **is** the package. There is **no** double nesting (i.e. not `ocr_classifier/ocr_classifier/`). It sits as a sibling of `ocr_mutasi/`, `ocr_slip/`, `ocr_match/` at the repo root:

```
ocr_classifier/                  ← this folder IS the package
├── __init__.py                  # __version__
├── config.py                    # pydantic-settings; reads the ROOT .env
├── models.py                    # enums + pydantic response models
├── ocr_client.py                # PaddleOCR upstream call + text extraction
├── llm_classifier.py            # Azure OpenAI, JSON-schema-constrained decision
├── pipeline.py                  # run() / run_batch() — orchestration
└── api.py                       # FastAPI app, endpoints, /upload page
```

- Import path `ocr_classifier.api:app` works because uvicorn is run from the repo root (which is on the Python path) — identical to `ocr_mutasi.api:app`.
- No per-service `requirements.txt` / `.env` / `.venv` / `README`: it reuses the repo-root ones, exactly like `ocr_mutasi`.
- Run from the repo root with: `uvicorn ocr_classifier.api:app --host 0.0.0.0 --port 8300`.

### 2.2 Ports

`ocr_mutasi` = 8000, `ocr_slip` = 8100, `ocr_match` = 8200, **`ocr_classifier` = 8300** (next free port). The port is supplied explicitly on the uvicorn command line, so the `APP_PORT=8000` value already present in the shared root `.env` (used by `ocr_mutasi`) does not clash; `config.py`'s `app_port` default is vestigial here.

### 2.3 High-level flow

```
            Browser / client
                  │
   POST /classify (multipart: file)
   POST /classify-batch (multipart: files[])
                  │
                  ▼
   ┌──────────────────────────────────┐
   │   ocr_classifier  (port 8300)     │
   │                                   │
   │   1. ocr_client.extract_text()    │──HTTP POST──▶ PaddleOCR service
   │      → OCR text + ocr audit       │              http://10.213.128.80:8090
   │                                   │              /predict/markdown
   │   2. llm_classifier.classify()    │──HTTPS─────▶ Azure OpenAI
   │      → {type, confidence, reason} │              (chat.completions, json_schema)
   │                                   │
   │   3. assemble ClassificationResult│
   └──────────────────────────────────┘
```

---

## 3. Data flow (single document)

```
PDF bytes (+ filename)
  → ocr_client.extract_text()
        POST {OCR_ENDPOINT_URL}?skip_orientation={OCR_SKIP_ORIENTATION}
        header:    X-API-Key: {OCR_API_KEY}
        multipart: file=<pdf bytes>
        → parse response (see §4)
        → join all block_content with "\n", normalize whitespace,
          truncate to MAX_CLASSIFY_CHARS
        → (text, OcrAudit{request_id, response_time_ms, char_count, page_count})
  → llm_classifier.classify(text)
        Azure chat.completions, temperature=0,
        response_format = json_schema → {document_type, confidence, reasoning}
  → ClassificationResult
```

**Batch** (`run_batch`): each file runs through the same single-document pipeline independently. OCR calls are issued concurrently with a cap of `BATCH_OCR_CONCURRENCY` (default 4), because each OCR call is slow (~10 s in the sample response). A per-file failure is captured in *that* file's `audit.errors` and never sinks the batch.

---

## 4. PaddleOCR upstream call & text extraction (`ocr_client.py`)

### 4.1 Request
Translated from the project owner's Postman collection:

- **Method/URL:** `POST http://10.213.128.80:8090/predict/markdown?skip_orientation=false`
- **Header:** `X-API-Key: bf7ed2b5…6572fc`
- **Body:** `multipart/form-data`, field `file` = the document bytes.

URL, key, and the `skip_orientation` flag come from config (§7). Uses `httpx.AsyncClient` with timeout `OCR_TIMEOUT_S`.

### 4.2 Response shape (observed sample)
```jsonc
{
  "response_code": 200,
  "error_message": null,
  "request_id": "9947da00-…",
  "timestamp": "2026-06-08T03:14:40…",
  "response_time_ms": 9725.01,
  "data": {
    "filename": "…_kk.pdf",
    "markdown": "<div …><img …/></div>\n",
    "json_result": {
      "input_path": "/tmp/…pdf",
      "page_index": 0,
      "model_settings": { … },
      "parsing_res_list": [
        { "block_label": "header", "block_content": "Foto Kartu Keluarga ", "block_bbox": [ … ] },
        { "block_label": "image",  "block_content": "KARTU KELUARGA No.3671… 02-04-2026", "block_bbox": [ … ] }
      ]
    }
  }
}
```

### 4.3 Text-extraction rules
1. The classification text is built from `data.json_result[*].parsing_res_list[*].block_content`.
2. **Page-shape normalization:** the sample shows `json_result` as a single object (one page, `page_index: 0`). A multi-page PDF may return a list of such objects. `ocr_client` normalizes `json_result` to a list (wrap a dict in a one-element list), then concatenates `block_content` across **all pages and all blocks**, joined with `"\n"`.
3. Whitespace is lightly normalized; the joined text is truncated to `MAX_CLASSIFY_CHARS` (default 8000). The distinguishing title/header tokens (`KARTU KELUARGA`, `KARTU TANDA PENDUDUK`, `SURAT KEPUTUSAN`, `SLIP GAJI`, `REKENING KORAN` / `MUTASI REKENING`) appear early, so truncation bounds tokens/cost without losing the signal.
4. **Fallbacks:** if `parsing_res_list` yields no text, strip HTML from `data.markdown` and use that. If text is still empty, short-circuit to `document_type: "unknown"`, `confidence: "low"`, an `audit.errors` note ("no text extracted from OCR"), and **skip** the LLM call.
5. `OcrAudit` carries `request_id`, `response_time_ms`, `extracted_char_count`, and `page_count` for debugging/traceability.

### 4.4 Upstream error mapping
Modeled on `ocr_match/upstream.py` (`UpstreamUnreachableError` / `UpstreamHttpError`):
- connect/DNS error → surfaced as HTTP `502` by the API layer;
- request timeout → `504`;
- OCR returns `>= 400` (or a non-200 `response_code` / non-null `error_message`) → `502`, with the upstream status/message in `detail`.

---

## 5. LLM classification (`llm_classifier.py`)

- **Client:** `AzureOpenAI` from the `openai` SDK, built from the shared `AZURE_OPENAI_*` settings, `temperature=0`, `timeout=LLM_REQUEST_TIMEOUT_S` — same construction as `ocr_mutasi/llm_classifier.py`.
- **Constrained output:** `response_format={"type": "json_schema", "json_schema": …}` so the reply deserializes straight into a pydantic model with zero string parsing. Schema:
  ```jsonc
  {
    "document_type": "ktp|kk|sk|slip|mutasi|unknown",   // enum, required
    "confidence":    "high|medium|low",                  // enum, required
    "reasoning":     "string"                            // one line, required
  }
  ```
- **System prompt** enumerates the five Indonesian document types and their distinguishing signals, and instructs `unknown` when nothing fits or the text is blank/illegible:
  | Label | Document | Signals |
  |---|---|---|
  | `ktp` | Kartu Tanda Penduduk (national ID) | `PROVINSI`, `NIK`, single-person identity, `Tempat/Tgl Lahir`, `Gol. Darah`, `Kewarganegaraan` |
  | `kk` | Kartu Keluarga (family card) | title `KARTU KELUARGA`, `No.` family-card number, `Nama Kepala Keluarga`, family-member table, `Hubungan Dalam Keluarga` |
  | `sk` | Surat Keputusan / Surat Keterangan (e.g. SK Pengangkatan) | `SURAT KEPUTUSAN` / `SURAT KETERANGAN`, `Menimbang`, `Memutuskan`, official decree/letter form |
  | `slip` | Slip Gaji (salary slip) | `SLIP GAJI`, `Gaji Pokok`, `Tunjangan`, `Potongan`, `Take Home Pay`, earnings/deductions |
  | `mutasi` | Mutasi Rekening / Rekening Koran (bank statement) | `REKENING KORAN` / `MUTASI REKENING`, dated rows with `Debit`/`Kredit`/`Saldo` |
  | `unknown` | none of the above / blank / illegible | — |
- **LLM-error handling:** on `APITimeoutError` / `APIError`, the result is returned with `document_type: "unknown"` and the error recorded in `audit.errors`. The service still answers `200`; a model hiccup does not 500 the request.

---

## 6. API surface (`api.py`)

| Method | Path | Body / params | Returns |
|---|---|---|---|
| POST | `/classify` | multipart `file` (+ `?include_text=true`) | one `ClassificationResult` |
| POST | `/classify-batch` | multipart repeated `files` (+ `?include_text=true`) | `{ "count": N, "results": [ ClassificationResult, … ] }` |
| GET | `/health` | — | `{ "status": "ok", "version": "…" }` |
| GET | `/` | — | `307` redirect to `/upload` |
| GET | `/upload` | — | minimal HTML multi-file upload page (mirrors the siblings) |
| GET | `/docs` | — | Swagger UI (with the OpenAPI 3.0.3 multi-file-upload patch copied from `ocr_mutasi/api.py`) |

- `include_text` (query, default **false**) controls whether the OCR text is echoed back in each result (it can be large).
- **Input:** PDFs are the primary input and match all five fixtures. The service forwards whatever bytes are uploaded to the OCR endpoint; it does not hard-reject non-PDF image types, since the OCR service accepts images too.
- **HTTP error codes:** `400` (no file / empty / wrong content), `413` (over `MAX_PDF_BYTES`, or batch over `MAX_FILES`), `502` (OCR unreachable / OCR 4xx-5xx), `504` (OCR timeout), `500` (unexpected server bug).

### 6.1 Response model (`models.py`)
```jsonc
// ClassificationResult
{
  "filename": "sample_kk.pdf",
  "document_type": "kk",            // DocumentType enum: ktp|kk|sk|slip|mutasi|unknown
  "confidence": "high",             // Confidence enum: high|medium|low
  "reasoning": "Title 'KARTU KELUARGA' and a family-member table.",
  "audit": {
    "ocr_request_id": "9947da00-…",
    "ocr_response_time_ms": 9725.0,
    "extracted_char_count": 1843,
    "page_count": 1,
    "errors": []                    // OCR / LLM problems recorded here (non-fatal)
  },
  "text": null                      // OCR text; populated only when include_text=true
}
```
Pydantic models: `DocumentType` (str enum), `Confidence` (str enum), `OcrAudit`, `ClassificationResult`, `BatchClassificationResponse { count, results }`.

---

## 7. Configuration

`config.py` is a pydantic-settings `Settings` reading the **repo-root `.env`** (`env_file=".env"`, `extra="ignore"`), with an `@lru_cache` singleton accessor — identical pattern to `ocr_mutasi/config.py`. It reuses the existing `AZURE_OPENAI_*` values and adds OCR settings.

New keys appended to the root `.env` / `.env.example` (the example file redacts the real key):
```
# --- PaddleOCR upstream (used by ocr_classifier) ---
OCR_ENDPOINT_URL=http://10.213.128.80:8090/predict/markdown
OCR_API_KEY=bf7ed2b5…6572fc
OCR_SKIP_ORIENTATION=false
OCR_TIMEOUT_S=120

# --- ocr_classifier limits ---
MAX_CLASSIFY_CHARS=8000
BATCH_OCR_CONCURRENCY=4
```
Reused existing keys: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_DEPLOYMENT`, `LLM_REQUEST_TIMEOUT_S`, `MAX_PDF_BYTES`. The classifier defines its own `app_port=8300` default and a `max_files=50` default.

---

## 8. Dependencies

The only new dependency is **`httpx`** (PaddleOCR upstream call + batch concurrency), added to the repo-root `requirements.txt`. Already present and reused: `fastapi`, `uvicorn[standard]`, `python-multipart`, `openai`, `pydantic`, `pydantic-settings`, `python-dotenv`.

---

## 9. Testing

- **Unit (no network):**
  - `ocr_client` text extraction — feed the captured sample OCR JSON into the parser and assert the joined `block_content`; cover the dict-vs-list `json_result` shapes, the empty-`parsing_res_list` → `markdown` fallback, and the empty-text short-circuit.
  - `llm_classifier` — mock the `AzureOpenAI` client; assert the schema→model mapping and the LLM-error → `unknown` path.
- **Smoke (live, opt-in):** a `scripts/smoke_classify.py` (or a pytest marked `network`) that runs the five built-in fixtures through the real OCR + LLM and asserts each returns its expected label. Skipped by default when there is no network access.

### 9.1 Ground-truth fixtures
The repo root already contains one labeled sample per type:

| Folder | Expected label |
|---|---|
| `classifier_ktp/` | `ktp` |
| `classifier_kk/` | `kk` |
| `classifier_sk/` | `sk` |
| `classifier_slip/` | `slip` |
| `classifier_mutasi/` | `mutasi` |

(Each folder holds one sample PDF; the smoke test globs `*.pdf` so exact filenames don't matter.)

---

## 10. Documentation

A dedicated `ocr_classifier/README.md` (endpoints, run command, `.env` keys, example `curl`), in addition to this spec. *(Revised during implementation: the repo-root `README.md` is in practice `ocr_mutasi`'s own doc — title "OCR Mutasi", structured around that one service — while the other siblings `ocr_match` and `ocr_slip` each carry their own folder README. A folder README is therefore the repo-consistent choice and keeps `ocr_mutasi`'s doc focused.)*

---

## 11. Open questions / future work

- **Confidence semantics:** `confidence` is the model's self-reported coarse signal (high/medium/low). It is advisory, not calibrated; downstream code should treat `low` / `unknown` as "needs a human look".
- **Multi-page `json_result`:** the exact multi-page response shape is unconfirmed; the normalization in §4.3 handles both dict and list defensively. To be verified against a real multi-page PDF during implementation.
- **Keyword shortcut (deferred):** a deterministic title-token pre-check (e.g. text contains `KARTU KELUARGA` → `kk`) could skip the LLM call for obvious cases. Out of scope now per instruction; noted as a possible later optimization.
