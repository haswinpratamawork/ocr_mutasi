# `ocr_classifier` — implementation plan

**Spec:** [2026-06-08-ocr-classifier-design.md](../specs/2026-06-08-ocr-classifier-design.md)
**Date:** 2026-06-08
**Target:** flat package `ocr_classifier/` at the repo root (mirrors `ocr_mutasi`), port 8300, shared root `.env` / `requirements.txt` / `.venv`.

Build bottom-up: config → models → OCR client → LLM classifier → pipeline → API → smoke/docs. Each phase ends with a concrete verification gate; do not advance until it passes. Phases 0–4 need **no network**; OCR/LLM-dependent verification (Phases 5–6 live checks) may require being on the corporate network/VPN, since `10.213.128.80` is a private address.

---

## Phase 0 — Scaffolding & configuration

**Files**
- `ocr_classifier/__init__.py` — `__version__ = "0.1.0"`.
- `ocr_classifier/config.py` — pydantic-settings `Settings` + `@lru_cache get_settings()`, copying the shape of `ocr_mutasi/config.py`. Fields:
  - reused: `azure_openai_endpoint`, `azure_openai_api_key`, `azure_openai_api_version` (default `2025-01-01-preview`), `azure_openai_deployment` (default `gpt-4.1-mini`), `llm_request_timeout_s` (default `120.0`), `max_pdf_bytes` (default `20_000_000`), `app_host`, `app_port` (default `8300`).
  - new: `ocr_endpoint_url`, `ocr_api_key`, `ocr_skip_orientation: bool = False`, `ocr_timeout_s: float = 120.0`, `max_classify_chars: int = 8000`, `batch_ocr_concurrency: int = 4`, `max_files: int = 50`.
  - `model_config`: `env_file=".env"`, `case_sensitive=False`, `extra="ignore"`.
- Edit `requirements.txt` (repo root): add `httpx>=0.27,<1`.
- Edit `.env` and `.env.example` (repo root): append the OCR block from spec §7 (`.env.example` redacts the key).

**Verify**
- `.venv/bin/python -c "from ocr_classifier.config import get_settings as g; s=g(); print(s.ocr_endpoint_url, s.app_port, s.batch_ocr_concurrency)"` run from repo root prints the configured values.
- `.venv/bin/pip install -r requirements.txt` succeeds (httpx installed).

---

## Phase 1 — Response models

**Files**
- `ocr_classifier/models.py`:
  - `DocumentType(str, Enum)` = `ktp | kk | sk | slip | mutasi | unknown`.
  - `Confidence(str, Enum)` = `high | medium | low`.
  - `OcrAudit` — `ocr_request_id: str | None`, `ocr_response_time_ms: float | None`, `extracted_char_count: int`, `page_count: int`, `errors: list[str] = []`.
  - `ClassificationResult` — `filename: str`, `document_type: DocumentType`, `confidence: Confidence`, `reasoning: str`, `audit: OcrAudit`, `text: str | None = None`.
  - `BatchClassificationResponse` — `count: int`, `results: list[ClassificationResult]`.

**Verify**
- Import and instantiate each model with dummy data in a one-liner; `.model_dump()` shape matches spec §6.1.

---

## Phase 2 — PaddleOCR client (`ocr_client.py`)

**Files**
- `ocr_classifier/ocr_client.py`:
  - Exceptions: `OcrUnreachableError`, `OcrTimeoutError`, `OcrHttpError(status_code, body)` — modeled on `ocr_match/upstream.py`.
  - **Pure helper** `extract_text_from_payload(payload: dict, max_chars: int) -> tuple[str, int]` returning `(text, page_count)`:
    - normalize `payload["data"]["json_result"]` to a list (wrap dict);
    - per page, collect `parsing_res_list[*].block_content`, join all with `"\n"`;
    - if empty, fall back to HTML-stripped `data.markdown`;
    - normalize whitespace, truncate to `max_chars`.
    - Keep this free of HTTP so it is unit-testable.
  - `async extract_text(file_bytes, filename) -> tuple[str, OcrAudit]`:
    - `httpx.AsyncClient(timeout=ocr_timeout_s)` POST to `ocr_endpoint_url`, `params={"skip_orientation": str(ocr_skip_orientation).lower()}`, `headers={"X-API-Key": ocr_api_key}`, `files={"file": (filename, file_bytes, "application/pdf")}`;
    - map `httpx.ConnectError` → `OcrUnreachableError`, `httpx.TimeoutException` → `OcrTimeoutError`, status `>= 400` → `OcrHttpError`; also treat a non-200 `response_code` / non-null `error_message` in the body as `OcrHttpError`;
    - build `OcrAudit` from `request_id`, `response_time_ms`, char count, page count.

**Verify (unit, no network)**
- `tests/test_ocr_client.py`: feed the spec §4.2 sample payload → assert the joined text contains `KARTU KELUARGA`; cover (a) `json_result` as dict, (b) as list of 2 pages → concatenated, (c) empty `parsing_res_list` → `markdown` fallback, (d) all-empty → `""` with `page_count` sane, (e) truncation at `max_chars`.
- `.venv/bin/pytest tests/test_ocr_client.py -q` green.

---

## Phase 3 — LLM classifier (`llm_classifier.py`)

**Files**
- `ocr_classifier/llm_classifier.py`:
  - `SYSTEM_PROMPT` — enumerate the five Indonesian doc types + signals from spec §5 table; instruct `unknown` for blank/illegible/none-of-the-above; instruct coarse `confidence`.
  - `_RESPONSE_SCHEMA` — strict json_schema for `{document_type(enum), confidence(enum), reasoning(string)}`.
  - `classify(text: str) -> tuple[DocumentType, Confidence, str, str | None]` returning `(type, confidence, reasoning, error)`:
    - build `AzureOpenAI(...)` from settings, `chat.completions.create(..., temperature=0, response_format={"type":"json_schema","json_schema":_RESPONSE_SCHEMA}, timeout=llm_request_timeout_s)`;
    - parse content into the enums;
    - on `APITimeoutError`/`APIError`/parse failure → return `(DocumentType.unknown, Confidence.low, "", "<error msg>")`.

**Verify (unit, no network)**
- `tests/test_llm_classifier.py`: monkeypatch the `AzureOpenAI` client so `chat.completions.create` returns a canned `kk` JSON → assert mapping; simulate `APITimeoutError` → assert `unknown` + error string. `pytest -q` green.

---

## Phase 4 — Pipeline (`pipeline.py`)

**Files**
- `ocr_classifier/pipeline.py`:
  - `async run(file_bytes, filename, include_text) -> ClassificationResult`:
    - call `ocr_client.extract_text`; if text empty → short-circuit `unknown`/`low`, audit note "no text extracted from OCR", skip LLM;
    - else call `llm_classifier.classify`; fold any LLM error into `audit.errors`;
    - attach `text` only when `include_text`.
  - `async run_batch(files, include_text) -> list[ClassificationResult]`:
    - `asyncio.Semaphore(batch_ocr_concurrency)`, `asyncio.gather` over `run(...)` per file;
    - each task wrapped so a per-file `Ocr*Error`/exception becomes a result with the error in `audit.errors` (batch never fails wholesale).

**Verify (unit, no network)**
- `tests/test_pipeline.py`: monkeypatch `ocr_client.extract_text` and `llm_classifier.classify`; assert (a) happy path result, (b) empty-text short-circuit skips LLM, (c) one failing file in a batch of 3 yields 3 results with the error isolated. `pytest -q` green.

---

## Phase 5 — API (`api.py`)

**Files**
- `ocr_classifier/api.py`:
  - `FastAPI(title="OCR Classifier", version=__version__, ...)`, `openapi_version="3.0.3"` + the `_custom_openapi` multi-file-upload patch copied from `ocr_mutasi/api.py`.
  - `POST /classify` — `file: UploadFile`, `include_text: bool = Query(False)`; size guard → `413` over `max_pdf_bytes`, empty/missing → `400`; call `pipeline.run`; map `OcrUnreachableError`/`OcrHttpError` → `502`, `OcrTimeoutError` → `504`.
  - `POST /classify-batch` — `files: list[UploadFile]`, `include_text` flag; `> max_files` → `413`; call `pipeline.run_batch`; return `BatchClassificationResponse`.
  - `GET /health` → `{"status":"ok","version":__version__}`.
  - `GET /` → 307 redirect to `/upload`; `GET /upload` → minimal HTML multi-file form (adapt the siblings' page).

**Verify (live-ish)**
- `.venv/bin/uvicorn ocr_classifier.api:app --host 127.0.0.1 --port 8300` starts clean.
- `curl -s localhost:8300/health` → `{"status":"ok",...}`; `/docs` renders and the upload field is a real file picker; `/` redirects to `/upload`.

---

## Phase 6 — Live smoke test & docs

**Files**
- `scripts/smoke_classify.py` (or `tests/test_smoke_live.py` marked `network`): POST each of the five fixtures (spec §9.1) to a running instance (or call `pipeline.run` directly) and assert `document_type` equals the expected label; print a pass/fail table.
- Edit root `README.md`: add an `ocr_classifier` section — purpose, run command, the new `.env` keys, `/classify` + `/classify-batch` `curl` examples, label list.

**Verify**
- With network access to `10.213.128.80:8090` and valid Azure creds, the smoke run labels all five fixtures correctly (`ktp/kk/sk/slip/mutasi`). If the OCR host is unreachable from the dev machine, document the expectation and defer the live run to an environment on the corp network.

---

## Phase 7 — Final pass & commit

- Run the full unit suite (`.venv/bin/pytest -q`) — green.
- Sanity re-read `config.py`/`api.py` for the shared-`.env` port nuance (8300 supplied on the CLI; `APP_PORT=8000` in `.env` is overridden).
- Commit: `feat: add ocr_classifier — PaddleOCR + LLM document-type classifier` (+ the `httpx`/`.env`/`README` edits).

---

## Risks / watch-items
- **Private OCR host.** `10.213.128.80` is internal; live OCR verification needs the corp network/VPN. Unit phases are fully offline.
- **Unconfirmed multi-page response shape.** §4.3 normalization handles dict and list; confirm against a real multi-page PDF when one is available.
- **LLM confidence is self-reported**, not calibrated — treat `low`/`unknown` as "needs human review", don't gate hard logic on `high`.
- **Shared `.env` drift.** Adding OCR keys to the root `.env` touches a file the siblings also read; `extra="ignore"` in each service's Settings keeps them tolerant of the new keys.
