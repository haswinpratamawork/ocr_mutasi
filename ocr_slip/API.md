# FastAPI Service

This service exposes the salary-slip parser through a local HTTP API.

## Start

```bash
python3 -m pip install -r requirements.txt
uvicorn api_app:app --host 127.0.0.1 --port 8000
```

Or:

```bash
./run_api.sh
```

Open the docs:

```text
http://127.0.0.1:8000/docs
```

## Endpoints

### `GET /`

Returns service metadata and endpoint paths.

### `GET /health`

Returns:

```json
{
  "status": "ok"
}
```

### `POST /parse`

Upload one or more PDF files.

Query parameters:

| Name | Values | Default | Description |
| --- | --- | --- | --- |
| `ocr` | `auto`, `never`, `always` | `auto` | Controls OCR fallback behavior. |

Example:

```bash
curl -X POST "http://127.0.0.1:8000/parse?ocr=auto" \
  -F "files=@/path/to/slip-1.pdf" \
  -F "files=@/path/to/slip-2.pdf"
```

Python client example:

```bash
python3 client_example.py /path/to/slip-1.pdf /path/to/slip-2.pdf
```

### `POST /parse-one`

Upload one PDF file. This endpoint is easiest to test from the browser docs.

Example:

```bash
curl -X POST "http://127.0.0.1:8000/parse-one?ocr=auto" \
  -F "file=@/path/to/slip.pdf"
```

Example response:

```json
{
  "source_file": "slip.pdf",
  "worker_name": "Example Worker",
  "institution_name": "PT Example Indonesia",
  "total_paid": 17878688,
  "pokok": 7027000,
  "tax": 4174172,
  "incentive": 24715105,
  "deduction": 13863417,
  "other_deduction": 9689245,
  "confidence_notes": [],
  "extraction_method": "pdf_text"
}
```

Example response:

```json
{
  "generated_at": "2026-06-01T08:07:20.325064+00:00",
  "document_count": 1,
  "totals": {
    "total_paid": 17878688,
    "pokok": 7027000,
    "tax": 4174172,
    "incentive": 24715105,
    "deduction": 13863417,
    "other_deduction": 9689245
  },
  "documents": [
    {
      "source_file": "sample_1.pdf",
      "worker_name": "Example Worker",
      "institution_name": "PT Example Indonesia",
      "total_paid": 17878688,
      "pokok": 7027000,
      "tax": 4174172,
      "incentive": 24715105,
      "deduction": 13863417,
      "other_deduction": 9689245,
      "confidence_notes": [],
      "extraction_method": "pdf_text"
    }
  ],
  "errors": []
}
```

## Privacy

Do not commit real salary slip PDFs to GitHub. The repository ignores
`input/*.pdf` and generated output folders by default.

## OCR Note

OCR fallback uses Apple Vision through PyObjC on macOS. Text-layer PDFs work on
other operating systems, but scanned PDFs need an OCR backend for that platform.
