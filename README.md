# OCR Services — Indonesian credit-document processing

A monorepo of small **FastAPI microservices** that read and understand the
documents in an Indonesian credit/loan application — bank statements, salary
slips, employment letters, and identity documents — and turn them into
structured, classified, cross-checked data.

Each service is a self-contained Python package that runs from this repo root.
They **share one virtualenv (`.venv`), one `.env`, and one `requirements.txt`**,
and each exposes a FastAPI app (Swagger at `/docs`) plus a browser upload page.

## Services

| Service | Port | What it does | Docs |
|---|---|---|---|
| **`ocr_classifier`** | 5001 | The front door: classify an uploaded document as `ktp` / `kk` / `sk` / `slip` / `mutasi` / `unknown`. Sends the file to a PaddleOCR service, then asks an LLM to label the text. | [`ocr_classifier/README.md`](ocr_classifier/README.md) |
| **`ocr_sk`** | 5002 | Parse **Surat Keterangan Kerja / Surat Pengangkatan** (employment letters) into structured fields. | [`ocr_sk/README.md`](ocr_sk/README.md) |
| **`ocr_slip`** | 5003 | Parse **salary slips** into worker / institution / take-home / pokok / tunjangan / potongan. | [`ocr_slip/README.md`](ocr_slip/README.md) |
| **`ocr_mutasi`** | 5004 | Parse **bank-statement mutations** (BCA, BRI, Mandiri, Permata, Sinarmas) and classify each credit (Gaji / THR / Bonus / Insentif / Lainnya). | [`ocr_mutasi/README.md`](ocr_mutasi/README.md) |
| **`ocr_match`** | 5005 | Reconcile salary slips against the bank's **`Gaji`** credit rows — confirms the declared income actually landed in the account. | [`ocr_match/README.md`](ocr_match/README.md) |

## How they fit together

```
                 ┌──────────────────┐
   any document  │  ocr_classifier  │  "what is this?"  → ktp / kk / sk / slip / mutasi
   ─────────────▶│      :5001       │
                 └──────────────────┘
                          │ route by type
        ┌─────────────────┼───────────────────┬──────────────┐
        ▼                 ▼                    ▼              ▼
  ┌───────────┐    ┌───────────┐        ┌───────────┐   (ktp/kk handled
  │  ocr_sk   │    │ ocr_slip  │        │ ocr_mutasi│    by classifier today)
  │  :5002    │    │  :5003    │        │  :5004    │
  └───────────┘    └─────┬─────┘        └─────┬─────┘
   employment           slips                 bank credits
   letters                └──────────┬─────────┘
                                     ▼
                              ┌───────────┐
                              │ ocr_match │  pairs each slip with its
                              │  :5005    │  matching Gaji credit row
                              └───────────┘
```

- **`ocr_classifier`** identifies a document so the caller can route it to the right parser.
- **`ocr_mutasi`** and **`ocr_slip`** are the two parsers feeding the income check.
- **`ocr_match`** is a thin orchestrator — it calls `ocr_slip` and `ocr_mutasi` over HTTP (never re-parses PDFs itself) and pairs slips with `Gaji` credits.
- **`ocr_sk`** parses employment letters; identity cards (`ktp` / `kk`) are recognised by `ocr_classifier`.

## Setup (once, shared by every service)

Assumes macOS / Apple Silicon with Homebrew Python 3.12; adapt the Python path for your OS.

```bash
# 1. one virtualenv + all dependencies for all services
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. one .env at the repo root
cp .env.example .env
$EDITOR .env     # fill in AZURE_OPENAI_* ; OCR_* and the service-URL registry have sane defaults
```

## Running a service

Every service has a **`run_api.sh`** that resolves the repo root, uses the
shared `.venv`, and binds the service's port — run it from anywhere; extra
flags pass through:

```bash
./ocr_classifier/run_api.sh    # http://localhost:5001
./ocr_sk/run_api.sh            # http://localhost:5002
./ocr_slip/run_api.sh          # http://localhost:5003
./ocr_mutasi/run_api.sh        # http://localhost:5004
./ocr_match/run_api.sh         # http://localhost:5005

./ocr_mutasi/run_api.sh --reload      # dev auto-reload
PORT=9000 ./ocr_sk/run_api.sh         # override host/port via HOST=/PORT=
```

Each service serves interactive API docs at `/docs` and a browser upload page
(`/upload`, or `/web` for `ocr_sk`). `ocr_match` needs its two upstreams
(`ocr_mutasi` on 5004 and `ocr_slip` on 5003) running.

## Running with Docker

Each service has a `Dockerfile` (built from the **repo root** so it can pull in
`ocr_common`), and `docker-compose.yml` runs all five together.

```bash
cp .env.example .env     # fill in AZURE_OPENAI_* and OCR_API_KEY (not baked into images)
docker compose up --build
```

That builds and starts all five on the same host ports as local runs
(classifier 5001, sk 5002, slip 5003, mutasi 5004, match 5005), each with a
`/health` healthcheck. Inside the compose network `ocr_match` reaches the others
by service name (`OCR_SLIP_URL=http://ocr_slip:5003`, `OCR_MUTASI_URL=http://ocr_mutasi:5004`).

Build/run a single service:

```bash
docker build -f ocr_classifier/Dockerfile -t ocr_classifier .
docker run --rm -p 5001:5001 --env-file .env ocr_classifier
```

Notes:
- Secrets are passed at runtime via `--env-file` / compose `env_file` — never baked into an image (and `.dockerignore` blocks `.env`, PDFs, and run artifacts).
- The PaddleOCR service (`OCR_ENDPOINT_URL`) is external; the Docker host must be able to reach it (e.g. on the corporate network).
- Each service's root (`/`) redirects to its API docs (`/docs`); the browser upload page stays at `/upload`.
- **Postman:** import [`ocr-services.postman_collection.json`](ocr-services.postman_collection.json) — every endpoint, with per-service base-URL variables defaulting to the local ports.

## Configuration (`.env`)

A single repo-root `.env` is shared by all services (see [`.env.example`](.env.example)):

| Keys | Used by | Purpose |
|---|---|---|
| `AZURE_OPENAI_*` | classifier, mutasi, slip, sk, match | Shared LLM credentials (Azure OpenAI) |
| `OCR_ENDPOINT_URL`, `OCR_API_KEY`, `OCR_SKIP_ORIENTATION`, `OCR_TIMEOUT_S` | classifier, slip, sk, mutasi | The shared PaddleOCR service (the single OCR backend) |
| `KETERANGAN_KERJA_FALLBACK` | sk | Enable the LLM text fallback |
| `OCR_*_URL` | match | Service-URL registry (`ocr_match` consumes `OCR_SLIP_URL` / `OCR_MUTASI_URL`) |
| `MAX_PDF_BYTES`, `MAX_FILES`, `MAX_CLASSIFY_CHARS`, `BATCH_OCR_CONCURRENCY`, `LLM_REQUEST_TIMEOUT_S` | various | Limits |

## Repo layout

```
ocr_mutasi/                 ← repo root (shared .venv, .env, requirements.txt, .gitignore)
├── ocr_classifier/         ← service package + Dockerfile + run_api.sh + README
├── ocr_sk/                 ← service package + web-ui/ + Dockerfile + run_api.sh + README
├── ocr_slip/               ← service package + Dockerfile + run_api.sh + README
├── ocr_mutasi/             ← service package + parsers/ + Dockerfile + run_api.sh + README
├── ocr_match/              ← service package + Dockerfile + run_api.sh + README
├── ocr_common/             ← shared PaddleOCR client (used by slip/sk/mutasi)
├── docs/
│   ├── architecture.md             ← ocr_mutasi internals & design
│   └── superpowers/specs|plans/    ← design specs & implementation plans
├── docker-compose.yml      ← run all five services together
├── .dockerignore
├── requirements.txt
├── .env.example
└── README.md               ← this file
```

Each service is a **flat package run from this root** (e.g. `ocr_mutasi.api:app`,
`ocr_slip.app:app`) — `run_api.sh` handles that for you. Running uvicorn from
*inside* a service folder fails with `ModuleNotFoundError`.

## External dependencies

- **Azure OpenAI** — the LLM used for classification, text-field fallback, and OCR transaction extraction.
- **PaddleOCR service** (`OCR_ENDPOINT_URL`) — the single OCR backend for the whole repo: `ocr_classifier` posts every document to it, and `ocr_slip` / `ocr_sk` / `ocr_mutasi` call it (via `ocr_common.paddle_ocr`) when their text-layer parser falls short. No local OCR engine is required.

## Data privacy

These documents contain PII (names, NIK, addresses, salaries, transactions).
`.gitignore` blocks `.env`, all `*.pdf`, virtualenvs, caches, and per-request
run outputs (`runs/`, `output*/`, `web_output/`). Only credit-transaction text
is sent to the LLM. Never commit real PDFs or secrets; rotate the Azure key if
one leaks.

## License

See [`LICENSE`](LICENSE).
