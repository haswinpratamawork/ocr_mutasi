# Salary Slip Parser

Parse salary-slip PDFs from an input folder with `pypdfium2`, then write:

- Raw extraction JSON with all text and lines extracted from each PDF.
- Salary summary JSON with paid salary, incentives/tunjangan, tax cutoffs, other cutoffs, worker name, and institution.

The project also includes a local FastAPI service so other people can use the
parser from their own machine.

## Install

This service is part of the `ocr_mutasi` monorepo and shares the **repo-root**
`.venv`, `requirements.txt`, and `.env` (its Azure credentials come from the
shared `AZURE_OPENAI_*` keys). From the repo root:

```bash
.venv/bin/pip install -r requirements.txt
```

Put local PDFs in `input/`. Real salary slips can contain private salary data,
so `input/*.pdf` is ignored by git.

## Run

```bash
python3 salary_slip_parser.py --input input --output output
```

OCR fallback is enabled by default. To control it:

```bash
python3 salary_slip_parser.py --input input --output output --ocr auto
python3 salary_slip_parser.py --input input --output output --ocr never
python3 salary_slip_parser.py --input input --output output --ocr always
```

The script writes:

- `output/extracted/<pdf-name>.json`
- `output/summary/<pdf-name>.json`
- `output/all_extracted.json`
- `output/salary_summary.json`

## Adapting to Other Slip Formats

The extraction layer is generic. The inference layer uses keyword lists in
`ParserConfig`. For a new salary slip format, create a JSON config with any
fields you want to override:

```json
{
  "net_pay_keywords": ["take home pay", "amount paid"],
  "deduction_sections": ["deductions", "withholding"],
  "incentive_keywords": ["allowance", "bonus", "commission"]
}
```

Then run:

```bash
python3 salary_slip_parser.py --input input --output output --config config.json
```

If a PDF is scanned and has no text layer, the script renders it with
`pypdfium2` and uses Apple Vision OCR on macOS. OCR results include a confidence
note such as `OCR fallback used via Apple Vision.`

See `WORKFLOW.md` for the optimized extraction flowchart and fallback strategy.

## Local Drag-and-Drop UI

Start the local web app:

```bash
python3 web_app.py
```

Then open:

```text
http://127.0.0.1:5050
```

Uploaded PDFs are processed through the same parser module. The web app stores
run output in `web_output/` and provides download links for `salary_summary.json`
and `all_extracted.json`.

To use another local port:

```bash
PORT=5051 python3 web_app.py
```

## Local FastAPI Service

`ocr_slip` is a package — run it from the **repo root** (where the shared
`.venv` and `.env` live), on port 8200:

```bash
.venv/bin/uvicorn ocr_slip.app:app --host 0.0.0.0 --port 8200 --reload
```

Open the interactive API docs:

```text
http://127.0.0.1:8200/docs
```

Parse one or more PDFs:

```bash
curl -X POST "http://127.0.0.1:8200/parse?ocr=auto" \
  -F "files=@/path/to/salary-slip.pdf"
```

For easiest browser testing in `/docs`, use:

```text
POST /parse-one
```

`POST /parse-one` accepts one PDF and shows a simpler file upload control.

The compact response includes:

- `total_paid`
- `pokok`
- `tax`
- `incentive`
- `deduction`
- `institution_name`
- `worker_name`

See `API.md` for endpoint details and example responses.

## Publishing to GitHub

This repository is configured to avoid committing private PDF inputs or
generated JSON outputs.

```bash
git init
git add .
git commit -m "Initial salary slip parser API"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

Before pushing, check what will be committed:

```bash
git status --short
```
