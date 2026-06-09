# OCR SK (Surat Keterangan Kerja)

Parse Indonesian **Surat Keterangan Kerja** / **Surat Pengangkatan** (employment
certificate / appointment letter) PDFs into structured JSON. Vendored from
[`hbridho/keterangan-kerja-ocr`](https://github.com/hbridho/keterangan-kerja-ocr)
and adapted to the monorepo's layout.

Pipeline: deterministic `pypdfium2` text extraction → rule-based classifier →
Tesseract OCR fallback for scanned PDFs → optional LLM text fallback for
missing fields.

It is a sibling of `ocr_mutasi` (8300), `ocr_slip` (8200), `ocr_match` (8400),
and `ocr_classifier` (8000), and runs on **port 8100**. It shares the repo-root
`.venv`, `requirements.txt`, and `.env` (its Azure credentials come from the
shared `AZURE_OPENAI_*` keys).

## Run

Run from the **repo root** (where the shared `.venv` and `.env` live), not from
inside `ocr_sk/` — it's a package, importable only from the root:

```bash
.venv/bin/uvicorn ocr_sk.app:app --host 0.0.0.0 --port 8100 --reload
```

- Browser upload page: <http://localhost:8100/web> (the bare URL redirects here)
- Swagger UI: <http://localhost:8100/docs>

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/parse` | multipart `files` (one or more PDFs) + optional `password` | `{ ok, needs_password, uploaded_files, summary, extracted, output_files }` |
| GET | `/health` | — | `{ status, parser_folder }` |
| GET | `/web` | — | the drag-and-drop upload UI (`web-ui/index.html`) |
| GET | `/` | — | redirect to `/web` |

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

# Tesseract OCR (shared with ocr_slip)
OCR_LANGUAGE=ind+eng
OCR_SCALE=3.0
OCR_PSM=6
OCR_OEM=1
OCR_BINARIZE=true
```

The OCR path uses the Indonesian Tesseract language data (`ind`); install it
(`brew install tesseract-lang`) for best accuracy. It degrades to English if
`ind` is not present.
