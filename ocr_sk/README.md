# OCR SK (Surat Keterangan Kerja)

Parse Indonesian **Surat Keterangan Kerja** / **Surat Pengangkatan** (employment
certificate / appointment letter) PDFs into structured JSON. Vendored from
[`hbridho/keterangan-kerja-ocr`](https://github.com/hbridho/keterangan-kerja-ocr)
and adapted to the monorepo's layout.

Pipeline: deterministic `pypdfium2` text extraction → rule-based classifier →
**PaddleOCR-service** fallback for scanned PDFs (shared `ocr_common` client) →
optional LLM text fallback for missing fields.

It is a sibling of `ocr_mutasi` (8300), `ocr_slip` (8200), `ocr_match` (8400),
and `ocr_classifier` (8000), and runs on **port 8100**. It shares the repo-root
`.venv`, `requirements.txt`, and `.env` (its Azure credentials come from the
shared `AZURE_OPENAI_*` keys).

## Run

Easiest — use the bundled `run_api.sh`. It resolves the repo root, uses the
shared `.venv`, and binds the right port; run it from anywhere and extra flags
pass through:

```bash
./ocr_sk/run_api.sh            # start on :8100
./ocr_sk/run_api.sh --reload   # dev auto-reload
PORT=9000 ./ocr_sk/run_api.sh  # override via HOST=/PORT=
```

Or run uvicorn directly, **from the repo root** (it's a package, importable only
from the root):

```bash
.venv/bin/uvicorn ocr_sk.app:app --host 0.0.0.0 --port 8100 --reload
```

- Browser upload page: <http://localhost:8100/upload> (the bare URL redirects here; `/web` is a legacy alias)
- Swagger UI: <http://localhost:8100/docs>

## Docker

Build this service's image (from the **repo root**, so it can include the shared
`ocr_common`) and run the container:

```bash
docker build -f ocr_sk/Dockerfile -t ocr_sk .
docker run --rm -p 8100:8100 --env-file .env ocr_sk
```

The API is then at <http://localhost:8100/docs> (browser upload page at `/upload`).
To run all five services together, use `docker compose up --build` from the repo
root — see the [root README](../README.md#running-with-docker).

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/parse` | multipart `files` (one or more PDFs) + optional `password` | `{ ok, needs_password, uploaded_files, summary, extracted, output_files }` |
| GET | `/health` | — | `{ status, parser_folder }` |
| GET | `/upload` | — | the drag-and-drop upload UI (`web-ui/index.html`); `/web` is a legacy alias |
| GET | `/` | — | redirect to `/upload` |

```bash
curl -s -F "files=@/path/to/surat-keterangan-kerja.pdf" \
  "http://localhost:8100/parse" | python -m json.tool
```

Parsed JSON is also written under `ocr_sk/output/` (`extracted.json`,
`summary.json`) — that directory is gitignored (it can contain PII).

## CLI

The parser also runs standalone (as a package module, from the repo root):

```bash
.venv/bin/python -m ocr_sk.extract_keterangan_kerja --help
```

## Configuration (repo-root `.env`)

Reuses the shared `AZURE_OPENAI_*` keys. Optional tunables (defaults shown):

```
# LLM text fallback for missing fields
KETERANGAN_KERJA_FALLBACK=true
```

When the text layer is too weak, the parser falls back to the shared
**PaddleOCR service** (configured by `OCR_ENDPOINT_URL` / `OCR_API_KEY` in the
root `.env`) via `ocr_common.paddle_ocr` — the same OCR backend every service
uses. No local OCR engine is required.

