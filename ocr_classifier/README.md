# OCR Classifier

A small FastAPI service that labels a document as one of five Indonesian
document types — **`ktp`**, **`kk`**, **`sk`**, **`slip`**, **`mutasi`** — or
**`unknown`**. It runs the uploaded file through the PaddleOCR markdown service,
joins the recognised text, and asks Azure OpenAI to classify it. It is a pure
classifier: it does not parse fields or route the document onward.

| Label | Document |
|---|---|
| `ktp` | Kartu Tanda Penduduk (national ID) |
| `kk` | Kartu Keluarga (family card) |
| `sk` | Surat Keputusan / Surat Keterangan (decree / letter, e.g. SK Pengangkatan) |
| `slip` | Slip Gaji (salary slip) |
| `mutasi` | Mutasi Rekening / Rekening Koran (bank statement) |
| `unknown` | none of the above / blank / illegible |

It is a sibling of `ocr_mutasi` (8300), `ocr_slip` (8200), and `ocr_match`
(8400), and runs on **port 8000**. It shares the repo-root `.env`,
`requirements.txt`, and `.venv`.

## How it works

```
PDF → POST {OCR_ENDPOINT_URL}?skip_orientation=…   (header X-API-Key, multipart file)
    → join data.json_result[*].parsing_res_list[*].block_content  (truncate to MAX_CLASSIFY_CHARS)
    → Azure OpenAI (json-schema-constrained) → {document_type, confidence, reasoning}
```

## Run

Run from the **repo root** (the parent of this folder — where the `.venv` and
`.env` live), NOT from inside `ocr_classifier/`. `ocr_classifier` is a flat
package at the repo root, so it's only importable when the repo root is the
working directory — same as `ocr_mutasi`. Running from inside this folder
fails with `ModuleNotFoundError: No module named 'ocr_classifier'`.

```bash
cd /path/to/ocr_mutasi          # the repo root, NOT ocr_mutasi/ocr_classifier
.venv/bin/uvicorn ocr_classifier.api:app --host 0.0.0.0 --port 8000 --reload
```

- Browser upload page: <http://localhost:8000/upload> (the bare URL redirects here)
- Swagger UI: <http://localhost:8000/docs>

## Endpoints

| Method | Path | Body / params | Returns |
|---|---|---|---|
| POST | `/classify` | multipart `file` (+ `?include_text=true`) | one result |
| POST | `/classify-batch` | multipart repeated `files` (+ `?include_text=true`) | `{count, results[]}` |
| GET | `/health` | — | `{status, version}` |
| GET | `/upload` | — | HTML upload page |

`include_text` (default `false`) echoes the extracted OCR text back in each
result. Batch files are classified concurrently (capped at
`BATCH_OCR_CONCURRENCY`); a per-file OCR/LLM failure is recorded in that
result's `audit.errors` and never sinks the batch.

### Examples

Each `classifier_*/` folder holds one sample PDF — substitute your own file paths.

```bash
# Single document
curl -s -F "file=@classifier_kk/sample_kk.pdf" \
  "http://localhost:8000/classify" | python -m json.tool

# A batch (repeat -F file= ... once per document)
curl -s \
  -F "files=@classifier_ktp/sample_ktp.pdf" \
  -F "files=@classifier_slip/sample_slip.pdf" \
  "http://localhost:8000/classify-batch?include_text=false" | python -m json.tool
```

Example single result:

```jsonc
{
  "filename": "sample_kk.pdf",
  "document_type": "kk",
  "confidence": "high",
  "reasoning": "Title 'KARTU KELUARGA' and a family-member table.",
  "audit": {
    "ocr_request_id": "9947da00-…",
    "ocr_response_time_ms": 9725.0,
    "extracted_char_count": 1843,
    "page_count": 1,
    "errors": []
  },
  "text": null
}
```

### HTTP errors

`400` empty upload · `413` over `MAX_PDF_BYTES` or batch over `MAX_FILES` ·
`502` OCR unreachable / OCR error · `504` OCR timeout · `500` unexpected.
(An LLM error is non-fatal: the result comes back as `unknown` with the error
in `audit.errors`.)

## Configuration (repo-root `.env`)

Reuses the existing `AZURE_OPENAI_*` keys and adds:

```
OCR_ENDPOINT_URL=http://10.213.128.80:8090/predict/markdown
OCR_API_KEY=<your-ocr-api-key>
OCR_SKIP_ORIENTATION=false
OCR_TIMEOUT_S=120
MAX_CLASSIFY_CHARS=8000
BATCH_OCR_CONCURRENCY=4
```

## Tests

```bash
# Offline unit tests (no network)
.venv/bin/python -m unittest discover -s tests -t . -v

# Live smoke test over the five built-in fixtures (needs corp-network access
# to the OCR host + valid Azure credentials)
.venv/bin/python scripts/smoke_classify.py
```
