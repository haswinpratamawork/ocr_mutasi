# OCR Mutasi

> Backend service that extracts transactions from Indonesian bank-statement PDFs and uses an LLM to classify each *credit* transaction as **Gaji** (fixed monthly salary), **THR** (religious-holiday allowance), **Bonus** (any `BONUS_*` label, annual or interim), **Insentif** (performance pay and work-related `TUNJANGAN <kind>` allowances), or **Lainnya** (other).

**Current version:** v0.10 — five supported banks, 5-category classifier (Gaji / THR / Bonus / Insentif / Lainnya), batch endpoint with cross-month classification, in-browser upload page, per-category `min` stat. v0.10 adds the Sinarmas "Tabungan" parser (6 bilingual columns; summary rows ABOVE the body; reverse-chronological order within the body; anchor at the BOTTOM of each visual block). See the change log in [architecture.md §19](../docs/architecture.md#19-change-log).

**Supported banks** (auto-detected from page 1 — no client flag needed):
- **BCA "Rekening Tahapan"** — 5-column layout, `DB` suffix marks debits.
- **BRI "BritAma"** — 6-column bilingual layout with separate Debet / Kredit columns.
- **Mandiri "Tabungan Mandiri" e-Statement** — 5-column bilingual layout, `+`/`-` prefix on the Nominal column signals credit/debit, Indonesian number format (`.` thousands, `,` decimals).
- **Permata "Rekening Koran"** — 6-column layout (Tgl Trx. / Tgl Valuta / Uraian Trx. / Debet / Kredit / Saldo), Indonesian number format like Mandiri, much wider page space than the other three (~2000pt vs ~600pt), holder name anchored to the "Kepada Yth" block.
- **Sinarmas "Tabungan"** — 6 bilingual columns (Date/Tanggal, Description/Keterangan, Detail, Debit/Debet, Credit/Kredit, Balance/Saldo). Layout is inverted vs the other four: period-summary rows (CLOSING BALANCE, MOVEMENT TOTALS) sit ABOVE the transaction body, column headers sit BELOW it, and rows are in reverse-chronological order with the anchor row (date + amount + balance) at the BOTTOM of each block. The parser walks rows top-to-bottom and reverses the result so the response stays forward-chronological. English number format.

**Two ways to test it:**
1. **Browser** — open <http://localhost:8300/upload>, pick one or many PDFs (multi-select via Cmd/Ctrl-click), get an accordion of classified credits.
2. **HTTP** — `POST /api/v1/mutations/extract` (single) or `/extract-batch` (multiple) with `multipart/form-data`. Schema explorer at <http://localhost:8300/docs>.

**Despite the name, no image OCR happens.** Mutasi PDFs ship with a clean embedded text layer; reading it with `pypdfium2` is faster, deterministic, and far more accurate than rasterising and OCR-ing. The "OCR" in the project name is historical. See [`docs/architecture.md`](../docs/architecture.md) for the full design and the alternatives rejected.

---

## Table of Contents

1. [Overview & Use Case](#1-overview--use-case)
2. [Quick Start (60 seconds)](#2-quick-start-60-seconds)
3. [Detailed Setup](#3-detailed-setup)
4. [Running the Service](#4-running-the-service)
5. [API Reference](#5-api-reference)
6. [Sample Responses](#6-sample-responses)
7. [Bank-Specific Notes](#7-bank-specific-notes)
8. [Real-World Validation](#8-real-world-validation)
9. [Performance & Cost](#9-performance--cost)
10. [Project Layout](#10-project-layout)
11. [Development Guide](#11-development-guide)
12. [Configuration Reference](#12-configuration-reference)
13. [Troubleshooting](#13-troubleshooting)
14. [FAQ](#14-faq)
15. [Limitations](#15-limitations)
16. [Roadmap](#16-roadmap)
17. [Data Privacy & Samples](#17-data-privacy--samples)

---

## 1. Overview & Use Case

A user uploads one or more monthly bank-statement PDFs. The backend extracts every transaction, then sends just the **credit rows** to an LLM (Azure OpenAI `gpt-4.1-mini`) to classify each as Gaji, THR, Bonus, Insentif, or Lainnya. The response is structured JSON with per-row classification, per-row confidence, and a year-level rollup when the batch endpoint is used.

**Typical flow:** the user selects a year of monthly mutations in your UI → the UI POSTs them as `multipart/form-data` to `/api/v1/mutations/extract-batch` → a single response gives the UI everything it needs to render income, allowances, and bonuses by category.

The pipeline at a glance:

```
PDF(s)  →  pypdfium2 text extraction  →  geometric table reconstruction (per bank)
       →  Transaction records  →  filter credit rows  →  Azure OpenAI (one call)
       →  JSON response
```

### Why two endpoints?

| Endpoint | When to use |
|---|---|
| `POST /api/v1/mutations/extract` | One PDF, one response. Good for ad-hoc inspection and debugging. |
| `POST /api/v1/mutations/extract-batch` | **Recommended for any "year of statements" workflow.** Sees every credit from every uploaded PDF in a *single* LLM call, so it catches recurring monthly payroll deposits whose individual rows look unremarkable. Real-world impact: 0 → 12 correct Gaji detections on the included 12-month BRI sample. |

### The 5 classification categories

Every credit row is assigned one of these. Definitions match Indonesian corporate-payroll conventions:

| Category | Definition | Common labels |
|---|---|---|
| **Gaji** | Fixed monthly salary that arrives on/near the same date with the same amount. | `GAJI`, `PAYROLL`, `SALARY`, `KR OTOMATIS GAJI`, `TRSF GAJI`, `SAP-DD` (SAP Direct Deposit), `PAYROLL-DEPOSIT` |
| **THR** | Tunjangan Hari Raya — religious-holiday allowance (Idul Fitri / Lebaran / Christmas). Paid 1–2× a year. | `THR`, `THR_Islam`, `THR_Idulfitri`, `THR_Lebaran`, `HARI RAYA`, `TUNJANGAN HARI RAYA` |
| **Bonus** | Any payment whose description starts with `BONUS_*`. Whether it's annual (`BONUS_POOL`, `BONUS_TAHUNAN`) or interim (`BONUS_INTERIM`), the `BONUS_` prefix is the company's own bonus-program label and all such rows are Bonus. | `BONUS_POOL`, `BONUS_TAHUNAN`, `BONUS_INTERIM`, `BONUS_YEARLY`, `ANNUAL_BONUS` |
| **Insentif** | Performance-tied pay **and** work-related `TUNJANGAN <kind>` allowances paid alongside Gaji. Anything an employee earns for performance or as a job perk lives here. | `INSENTIF`, `INCENTIVE`, `KOMISI`, `COMMISSION`, `ECUTI` (extra-cuti payout), `TUNJANGAN TRANSPORT`, `TUNJANGAN MAKAN`, `TUNJANGAN PULSA`, `TUNJANGAN KELUAR KOTA`, `TUNJANGAN KESEHATAN`, `TUNJANGAN ANAK`, `TUNJANGAN ISTRI` — i.e. any `TUNJANGAN <kind>` *except* `TUNJANGAN HARI RAYA` |
| **Lainnya** | Anything the rules above don't match: P2P transfers from a person's name, refunds, interest, sale proceeds, self-transfers, reimbursements. | `Transfer Dari <name>`, `BIF TRANSFER DR <name>`, `BUNGA TABUNGAN`, refund descriptors |

### Decision rules (applied in order, first match wins)

The LLM follows these six rules verbatim from the system prompt. Each row's `reason` field cites the rule that fired, so misclassifications are easy to diagnose.

1. Description contains `BONUS_…` or `BONUS ` → **Bonus**
2. Description contains `THR` / `HARI RAYA` / `TUNJANGAN HARI RAYA` → **THR** *(must run before rule 3 so the generic-tunjangan catchall doesn't swallow it)*
3. Description contains `ECUTI` / `INSENTIF` / `INCENTIVE` / `KOMISI` / `COMMISSION`, **or** any `TUNJANGAN <kind>` other than Hari Raya (transport, makan, pulsa, keluar kota, kesehatan, …), **or** `LLG-DEUTSCHE BANK` / any `LLG ` prefix (Lalu Lintas Giro — BI bulk-clearing channel used for allowance disbursement) → **Insentif**. *Because this rule fires before rule 4, a mixed label like `KR OTOMATIS LLG-DEUTSCHE BANK | PT <X>` resolves to Insentif, not Gaji.*
4. Description contains an explicit payroll-disbursement keyword **and a corporate sender** (`PT <X>`, `<X> PT`, `<X> INDO`, `<X> BSD`, `KASTARA <X>`, …) → **Gaji**. Three flavours:
   - employer-facing labels: `GAJI` / `PAYROLL` / `SALARY` / `TRSF GAJI` / `PAYROLL-DEPOSIT` / `SALARY-CRDT`
   - Indonesian bank bulk-payroll product labels: `SAP-DD` (SAP Direct Deposit), `KR OTOMATIS` (BCA auto-credit *when NOT accompanied by an `LLG` label*), `SMEMFTS` (BCA SME Mass Funds Transfer Service — the primary salary channel)
   - **Professional-fee / honorarium labels** *(new in v0.8)* — payment FOR work done where the description names the kind of work: `FEE DOKTER`, `FEE DRG`, `FEE NOTARIS`, `FEE KONSULTAN`, `FEE [profession]`, `HONOR`, `HONORARIUM`, `JASA <name>`, `RETAINER`. Example matches: `TRSF E-BANKING CR <ref> | FEE DOKTER | PT KLINIK CONTOH JAKARTA` → Gaji.

   **Hard exclusion for rule 4 (overrides any match above):** descriptions containing `CASHBACK`, `REFUND`, `REIMBURSE`, `BUNGA` (bank interest), `TAX REFUND`, or `PROMO` always go to Lainnya — these are merchant/bank disbursements, not employer payments, even when the apparent "sender" looks corporate (e.g. `KR OTOMATIS TRF KOLEKTIF | CASHBACK QRIS BCA | DI MERCHANT XYZ` → Lainnya).
5. *(Batch endpoint only)* No label match, but **the same corporate sender** (`PT <X>`, `<X> INDO`, …) appears in **multiple uploaded months** → **Gaji**. Sender consistency is the cross-month signal — real salaries vary in amount month-to-month (overtime, deductions, prorated months, bundled THR/bonus), so amount equality is NOT required.
6. Otherwise → **Lainnya**

The full prompt text lives in `ocr_mutasi/llm_classifier.py` (`SYSTEM_PROMPT` for single-PDF, `BATCH_SYSTEM_PROMPT` for batch).

---

## 2. Quick Start (60 seconds)

Assumes macOS / Apple Silicon with Homebrew's Python 3.12 installed; adapt the Python path for your OS.

```bash
# 1. Create the venv and install dependencies
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Configure Azure OpenAI credentials
cp .env.example .env
$EDITOR .env          # paste real AZURE_OPENAI_* values

# 3. Start the server (run_api.sh cd's to the repo root and binds :8300)
./ocr_mutasi/run_api.sh --reload

# 4. (in another shell) Hit the recommended batch endpoint with a folder of PDFs
curl -X POST $(for f in mutasi_haswin/Mutasi_*.pdf; do echo -n "-F files=@$f "; done) \
  "http://127.0.0.1:8300/api/v1/mutations/extract-batch" \
  | jq .audit.category_totals
```

Expected output for the included 12-month BRI sample:

```json
{
  "Gaji":     { "count": 12, "sum": 120000000.0, "min": 9500000.0 },
  "THR":      { "count": 1,  "sum": 23000000.0,  "min": 23000000.0 },
  "Bonus":    { "count": 2,  "sum": 44000000.0,  "min": 7000000.0 },
  "Insentif": { "count": 2,  "sum": 21000000.0,  "min": 10000000.0 },
  "Lainnya":  { "count": 81, "sum": 50000000.0,  "min": 10000.0 }
}
```

Browse the interactive OpenAPI docs at <http://127.0.0.1:8300/docs>.

---

## 3. Detailed Setup

### 3.1 Prerequisites

- **Python 3.11 or 3.12.** Project was developed against 3.12.4. Python 3.10 should work but isn't tested.
- **No system OCR dependency.** `pypdfium2` ships precompiled wheels for macOS / Linux / Windows; nothing to install via Homebrew or apt.
- **Network access to Azure OpenAI** for classification. The pipeline still extracts without it — credits are returned with `category: null` and an entry in `audit.classifier_errors`.

Platform-specific Python installation hints:

| OS | Recommended path |
|---|---|
| macOS (Homebrew) | `brew install python@3.12` → `/opt/homebrew/bin/python3.12` |
| macOS (pyenv) | `pyenv install 3.12.4 && pyenv shell 3.12.4` |
| Ubuntu / Debian | `sudo apt install python3.12 python3.12-venv` |
| Windows | Install from <https://python.org> (3.12.x), then use `py -3.12` |

### 3.2 Create the virtual environment

From the project root (the directory containing `requirements.txt`):

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

The `.venv/` directory is gitignored.

### 3.3 Configure secrets

```bash
cp .env.example .env
```

Edit `.env` with the real Azure OpenAI values. Only the four `AZURE_OPENAI_*` variables are required; the rest are tunable with sensible defaults (see §12 for the full reference).

```dotenv
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-api-key>
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini
```

> **Security:** `.env` is gitignored. Never commit real keys. If a key leaks, rotate it via the Azure portal immediately.

### 3.4 Get a sample PDF

**The repo does not ship any bank-statement PDFs** — they're personal financial data and excluded via `.gitignore` (see [§17 below](#17-data-privacy--samples)). To exercise the smoke tests and the upload page, drop one of your own into the project root:

```bash
# pick any BCA "Rekening Tahapan", BRI "BritAma", Mandiri "Tabungan Mandiri", Permata "Rekening Koran", or Sinarmas "Tabungan" PDF
cp ~/Downloads/your_statement.pdf ./sample.pdf
```

The smoke tests in §11.1 reference filenames like `contoh_mutasi.pdf` and `sample_mandiri.pdf`; rename your file accordingly or edit the path in the snippet.

### 3.5 Verify the install

```bash
.venv/bin/python -c "
from ocr_mutasi.pipeline import run
from pathlib import Path
r = run(Path('contoh_mutasi.pdf').read_bytes(), classify=False)
print(f'OK — {r.account.bank} {r.audit.rows_detected} tx '
      f'(DB={r.audit.debit_count}, CR={r.audit.credit_count})')
"
```

Expected: `OK — BCA 134 tx (DB=131, CR=3)`.

---

## 4. Running the Service

### 4.1 Development (auto-reload)

```bash
./ocr_mutasi/run_api.sh --reload     # shortcut — cd's to repo root, binds :8300
# equivalent explicit command:
.venv/bin/uvicorn ocr_mutasi.api:app --host 0.0.0.0 --port 8300 --reload
```

`--reload` watches your source files and restarts on changes. Use only in dev.
`run_api.sh` accepts `HOST=` / `PORT=` overrides and passes extra flags through.

### 4.2 Production-like

```bash
.venv/bin/uvicorn ocr_mutasi.api:app --host 0.0.0.0 --port 8300 --workers 4
```

`--workers 4` runs four uvicorn worker processes behind one socket — useful when the LLM call is the bottleneck (which it is) and you want to handle multiple uploads concurrently.

### 4.3 Behind a gateway

The service has **no authentication and no rate limiting**. Deploy it behind a gateway (NGINX, Cloud Run, API Gateway, etc.) that handles both.

### 4.4 Run with Docker

Build this service's image (from the **repo root**, so it can include the shared `ocr_common`) and run the container:

```bash
docker build -f ocr_mutasi/Dockerfile -t ocr_mutasi .
docker run --rm -p 8300:8300 --env-file .env ocr_mutasi
```

The API is then at <http://localhost:8300/docs> (browser upload page at `/upload`). To run all five services together, use `docker compose up --build` from the repo root — see the [root README](../README.md#running-with-docker).

---

## 5. API Reference

Three endpoints. All payloads and responses are JSON. Browse the live OpenAPI explorer at `/docs`.

### 5.1 `GET /health`

Liveness probe.

| | |
|---|---|
| Request body | — |
| Response | `200 OK` — `{"status": "ok", "version": "<v>"}` |

```bash
curl http://127.0.0.1:8300/health
# {"status":"ok","version":"0.1.0"}
```

### 5.2 `POST /api/v1/mutations/extract-batch` *(recommended for a year of statements)*

Accept **multiple PDFs in one request** and run **one** cross-month LLM classification across every credit row from every file. This is the only way to detect recurring monthly payroll deposits — the strongest Gaji signal. On the included 12-month BRI sample this endpoint catches 12 Gaji rows that the single-PDF endpoint misses entirely.

| | |
|---|---|
| Request | `multipart/form-data` with **one or more `files` fields**, each a PDF — the field name is `files` (plural) and is repeated per file |
| Query params | `classify=true\|false` (default `true`). `false` skips the LLM call. |
| Response | `200 OK` — `BatchExtractionResponse` (see §6.2) |
| Errors | `400` (no files / wrong content-type / empty), `413` (too large), `422` (unparseable PDF / unsupported bank / no transactions), `500` (server bug) |

#### How to send multiple files

The trick is that **`files` is repeated once per file** in the multipart body — not one field with a list value. Every transport has its own way to express this:

##### a) Built-in upload page — `http://localhost:8300/upload` *(recommended for testing)*

The simplest path. Open the URL in a browser, click **Choose PDFs**, multi-select files in the OS dialog (Cmd-click on macOS, Ctrl-click on Windows/Linux, Shift-click to range-select), and click **Extract**. The bare URL `http://localhost:8300/` redirects here too.

After the response arrives, the page renders a **per-category accordion**:

```
▾ Gaji         12 tx · min Rp 9,500,000.00      Rp 120,000,000.00
   Date        Source file              Amount       Description        Conf.  Reason
   2025-05-23  Mutasi_Mei_2025.pdf      Rp 9.5M      SAP-DD TRANSACTION 0.95   recurs monthly…
   2025-06-25  Mutasi_Juni_2025.pdf     Rp 9.5M      SAP-DD TRANSACTION 0.95   …
   …
▸ THR           1 tx · min Rp 23,000,000.00     Rp 23,000,000.00    (click row to expand)
▸ Bonus         2 tx · min Rp 7,000,000.00      Rp 44,000,000.00
▸ Insentif      2 tx · min Rp 10,000,000.00     Rp 21,000,000.00
▸ Lainnya      81 tx · min Rp 10,000.00         Rp 50,000,000.00
```

Gaji / THR / Bonus / Insentif expand by default; Lainnya stays collapsed (usually noisy). Category names are colour-coded. The full raw JSON is available behind a collapsible toggle.

This page bypasses Swagger UI's array-renderer (which insists on one file slot per array item with an "Add item" button — a Swagger UI design choice that no OpenAPI schema can work around).

##### b) Swagger UI (`http://localhost:8300/docs`)

Swagger UI renders `files: List[UploadFile]` as one file-picker row per array item, with an **Add item** button to add more rows. Each row holds one file. So for 12 monthly statements: click **Add item** 11 times to get 12 slots, then pick one PDF per slot. This is awkward — prefer `/upload` (above) or curl (below) for batch testing. Swagger UI is best for inspecting the schema and trying single-PDF endpoints.

##### b) `curl` — repeat `-F files=@…` once per file

```bash
# Three files, one at a time
curl -X POST \
  -F files=@Mutasi_Januari_2026.pdf \
  -F files=@Mutasi_Februari_2026.pdf \
  -F files=@Mutasi_Maret_2026.pdf \
  http://localhost:8300/api/v1/mutations/extract-batch

# Or expand a whole folder with the shell
curl -X POST \
  $(for f in mutasi_haswin/Mutasi_*.pdf; do echo -n "-F files=@$f "; done) \
  http://localhost:8300/api/v1/mutations/extract-batch \
  -o batch.json
```

##### c) Python — `httpx` with a list of `("files", ...)` tuples

```python
import glob, httpx

paths = sorted(glob.glob("mutasi_haswin/Mutasi_*.pdf"))
# IMPORTANT: every tuple's first element is the literal string "files"
files = [
    ("files", (p.split("/")[-1], open(p, "rb"), "application/pdf"))
    for p in paths
]
r = httpx.post(
    "http://localhost:8300/api/v1/mutations/extract-batch",
    files=files,
    timeout=120,
)
print(r.status_code, r.json()["audit"]["category_totals"])
```

##### d) Python — `requests` (same shape)

```python
import glob, requests

paths = sorted(glob.glob("mutasi_haswin/Mutasi_*.pdf"))
files = [("files", (p.split("/")[-1], open(p, "rb"), "application/pdf"))
         for p in paths]
r = requests.post(
    "http://localhost:8300/api/v1/mutations/extract-batch",
    files=files, timeout=120,
)
print(r.json()["audit"]["category_totals"])
```

##### e) JavaScript / `fetch` (browser or Node)

```javascript
const fd = new FormData();
for (const file of fileInput.files) {     // <input type="file" multiple>
  fd.append("files", file, file.name);    // same field name "files" each time
}
const r = await fetch("/api/v1/mutations/extract-batch", { method: "POST", body: fd });
const data = await r.json();
console.log(data.audit.category_totals);
```

> Note for the frontend: use `<input type="file" multiple accept="application/pdf">` so the user can multi-select PDFs from one dialog.

#### Validated behaviour

```
POST /extract-batch with 3 BRI PDFs (no LLM):
  → 200 OK in 0.37 s
  → files in response: ['Mutasi_Januari_2026.pdf', 'Mutasi_Februari_2026.pdf', 'Mutasi_Maret_2026.pdf']
  → per-file tx counts: Januari=76, Februari=63, Maret=103
  → audit.transactions_total=242, audit.credits_total=19
```

### 5.3 `POST /api/v1/mutations/extract`

Extract and (optionally) classify a single PDF.

| | |
|---|---|
| Request | `multipart/form-data` with one `file` field |
| Query params | `classify=true\|false` (default `true`) |
| Response | `200 OK` — `ExtractionResponse` (see §6.1) |
| Errors | Same status codes as the batch endpoint |

```bash
curl -X POST -F "file=@contoh_mutasi.pdf" \
  "http://127.0.0.1:8300/api/v1/mutations/extract?classify=true" \
  | jq .
```

### 5.4 Error response format

All non-2xx responses are JSON of the form:

```json
{"detail": "<one-line human-readable message>"}
```

Examples:

| Status | Example body |
|---|---|
| `400` | `{"detail": "Upload must be a PDF."}` |
| `413` | `{"detail": "'Mutasi_April_2026.pdf': exceeds 20000000 bytes"}` |
| `422` | `{"detail": "Could not read PDF: Failed to load document (PDFium: Data format error)."}` |
| `422` | `{"detail": "PDF doesn't match any known bank layout (supported: BCA Rekening Tahapan, BRI BritAma, Mandiri Tabungan, Permata Rekening Koran, Sinarmas Tabungan)."}` |
| `422` | `{"detail": "No transactions detected — is this a supported bank PDF?"}` |
| `500` | `{"detail": "Internal error while parsing PDF"}` |

The `200 OK` response always includes an `audit` block. Even on a successful response, check `audit.classifier_errors` (LLM failure) and `audit.balance_warnings` (suspect arithmetic) to judge result quality.

---

## 6. Sample Responses

### 6.1 `/extract` response (single PDF)

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
      "tanggal": "2026-04-01",            // ISO 8601 date
      "keterangan": "TRANSAKSI DEBIT TGL: 01/04 | QR 008 | 00000.00DAMRI-0364",
      "cbg": null,                        // BCA leaves this empty; BRI puts the User-ID here
      "amount": 25000.0,
      "type": "DB",                       // "DB" debit or "CR" credit
      "saldo": 1475000.00,                // null when the statement omits it
      "page": 1                           // 1-indexed source page
    }
    // … all transactions in source order
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
      "category": "Lainnya",              // Gaji | THR | Bonus | Insentif | Lainnya | null
      "confidence": 0.7,                  // 0..1; null if the LLM call failed
      "reason": "No salary keywords; self-transfer."
    }
  ],
  "audit": {
    "pages_processed": 11,
    "rows_detected": 134,
    "credit_count": 3,
    "debit_count": 131,
    "balance_warnings": [],               // empty = arithmetic reconciles
    "parse_warnings": [],                 // empty = every row parsed cleanly
    "classifier_errors": []               // empty = LLM call succeeded
  }
}
```

### 6.2 `/extract-batch` response (12 PDFs)

```jsonc
{
  "files": [
    {
      "filename": "Mutasi_Mei_2025.pdf",
      "account":      { "bank": "BRI", "no_rekening": "123456789012345", "nama": "BUDI SANTOSO", "periode": "01/05/25 - 31/05/25", "mata_uang": "IDR" },
      "transactions": [ /* every transaction in this file */ ],
      "audit":        { "pages_processed": 4, "rows_detected": 76, "credit_count": 8, "debit_count": 68, "balance_warnings": [], "parse_warnings": [], "classifier_errors": [] }
    }
    // … one entry per uploaded file, in upload order
  ],
  "credits": [
    {
      "source_file": "Mutasi_Mei_2025.pdf",
      "tanggal": "2025-05-23",
      "keterangan": "SAP-DD TRANSACTION",
      "cbg": "8888047",
      "amount": 9500000.0,
      "type": "CR",
      "saldo": 11041425.84,
      "page": 1,
      "category": "Gaji",
      "confidence": 0.95,
      "reason": "SAP-DD TRANSACTION recurs monthly with similar amount and timing, strong payroll signal."
    }
    // … all credit rows across all files
  ],
  "audit": {
    "files_processed": 12,
    "transactions_total": 951,
    "credits_total": 98,
    "classifier_errors": [],
    "category_totals": {
      "Gaji":     { "count": 12, "sum": 120000000.0, "min":  9500000.0 },
      "THR":      { "count":  1, "sum":  23000000.0, "min": 23000000.0 },
      "Bonus":    { "count":  2, "sum":  44000000.0, "min":  7000000.0 },
      "Insentif": { "count":  2, "sum":  21000000.0, "min": 10000000.0 },
      "Lainnya":  { "count": 81, "sum":  50000000.0, "min":    10000.0 }
    }
  }
}
```

---

## 7. Bank-Specific Notes

### 7.1 BCA "Rekening Tahapan"

- **Detection token:** `REKENING TAHAPAN` on page 1.
- **Columns:** `TANGGAL / KETERANGAN / CBG / MUTASI / SALDO`.
- **Date format:** `DD/MM`. The year comes from the `PERIODE` header (e.g. `APRIL 2026`).
- **Debit vs credit:** the `MUTASI` cell ends with `DB` for debits; credits have **no suffix**. The KETERANGAN main label (`TRANSAKSI DEBIT`, `TRSF E-BANKING DB`, `BI-FAST CR`) is used as a cross-check.
- **Multi-line transactions:** each transaction's `KETERANGAN` typically spans 1–6 lines (`TGL: ...`, `QR 008`, `00000.00DAMRI-0364`, etc.). These are joined with ` | ` in the response.
- **SALDO not always present:** the running balance is only printed on some rows (often the day's last). Cross-row balance check works around this with a running balance maintained over every transaction; see [architecture §11.1](../docs/architecture.md#111-balance-continuity).

### 7.2 BRI "BritAma"

- **Detection tokens:** `LAPORAN TRANSAKSI FINANSIAL` or `BRITAMA` on page 1.
- **Columns:** `Tanggal Transaksi / Uraian Transaksi / Teller / Debet / Kredit / Saldo` (Indonesian above English subheader).
- **Date format:** `DD/MM/YY HH:MM:SS`. The 2-digit year is expanded to `2000 + yy`.
- **Debit vs credit:** **two separate columns**. The unused one holds a `0.00` placeholder. `type = "CR"` when `Kredit > 0`, `"DB"` when `Debet > 0`.
- **Teller / User ID:** surfaced in the `Transaction.cbg` field (closest analogue to BCA's CBG).
- **Summary block trim:** the last page has a `Saldo Awal / Total Transaksi Debet / Total Transaksi Kredit / Saldo Akhir` summary. This is explicitly trimmed before parsing so its numbers don't pollute the last transaction.

### 7.3 Mandiri "Tabungan Mandiri" e-Statement

- **Detection tokens:** `Tabungan Mandiri` or `Menara Mandiri` on page 1.
- **Columns:** `No / Tanggal-Date / Keterangan-Remarks / Nominal (IDR)-Amount / Saldo (IDR)-Balance` (Indonesian above English subheader).
- **Date format:** `DD MMM YYYY` with the time on a separate line (`08:56:51 WIB`). English month abbreviations (`Apr`, `May`) are primary; Indonesian (`Mei`, `Agt`, `Okt`, `Des`) also accepted.
- **Number format is INVERTED** vs BCA/BRI: Mandiri uses `.` as thousands separator and `,` as decimal (e.g. `6.000.000,00` = 6,000,000.00). The parser uses a Mandiri-specific `_parse_id_amount` helper.
- **Debit vs credit:** sign-prefixed on the Nominal column. `+` = credit (incoming), `-` = debit (outgoing). No separate columns, no DB/CR suffix.
- **Multi-line transactions:** each transaction spans 3–5 visual lines (main label, date, anchor line with No+values, time, optional description continuation). The grouper packs consecutive lines within ~18 pt vertical gap into one transaction.
- **Account name in the header** may be split across two chunks (`BUDI` on one line, `SANTOSO` on the next) — the field extractor concatenates them.

### 7.4 Permata "Rekening Koran"

- **Detection tokens:** `PERMATABANK.COM`, `PT BANK PERMATA`, or `PERMATABANK` on page 1 (the URL/phone block is the cleanest disambiguator; the title "Rekening Koran" alone could collide with other banks' future statements).
- **Columns:** `Tgl Trx. / Tgl Valuta / Uraian Trx. / Debet / Kredit / Saldo` (6 columns — same shape as BRI but using Indonesian-format numbers).
- **Page scale is ~3× larger** than BCA/BRI/Mandiri: Permata renders into a ~2000pt-wide coordinate space (vs ~600pt for the others). The line-clustering tolerance and column-boundary heuristics in `parsers/permata.py` are scale-relative, not hard-coded constants.
- **Number format:** Indonesian, like Mandiri (`.` thousands, `,` decimals; e.g. `6.000.000,00`). Uses the shared `_parse_id_amount` helper.
- **Debit vs credit:** **two separate columns**, like BRI — the unused one is blank rather than `0.00`. `type` is set from whichever column has a value.
- **Multi-line `Uraian Trx.`** is the norm: most rows carry 2–4 continuation lines (recipient detail, BIFAST reference, time-of-day, payment reference number). The grouper joins them with ` | ` so they read naturally.
- **Holder name** is extracted from the "Kepada Yth" address block on page 1, filtering out lines that look like a street address (e.g. starting with `JL`, `RT`, `KEL`, `KEC`, postal codes).

### 7.5 Sinarmas "Tabungan"

- **Detection tokens:** `SINARMAS` or `BANK SINARMAS` on page 1.
- **Columns (six, bilingual):** `Date / Tanggal`, `Description / Keterangan`, `Detail` (counterparty), `Debit / Debet`, `Credit / Kredit`, `Balance / Saldo`. English label is the upper line, Indonesian the lower.
- **Page layout is inverted vs every other supported bank:** period-summary rows (`CLOSING BALANCE`, `MOVEMENT TOTALS`) sit ABOVE the transaction body; column headers sit BELOW the body; the account-info block (period, holder name, account no., currency, category) sits at the very bottom of page 1.
- **Reverse-chronological order WITHIN the body:** the most recent transaction is at the top; the opening balance (`BALANCE PERIOD START`) is at the bottom. The parser collects rows in source order then REVERSES the result so the response stays forward-chronological like every other bank.
- **Block anchor is at the BOTTOM:** each transaction's anchor row (date + amount + balance) sits at the LOWEST y of its visual block; continuation lines (counterparty bank, counterparty account number, QR-merchant code, branch suffix) sit ABOVE it. We accumulate continuations as we descend and attach them to the next anchor below.
- **English number format** (`,` thousands, `.` decimals) — same as BCA/BRI.
- **Leading-digit chunk split:** long debit amounts can render as a single leading digit chunk plus the rest, e.g. a leading `9` chunk followed by `9,000,000.00` → `99,000,000.00`. The amount-column joiner uses NO-SPACE concat so they merge cleanly before parsing.
- **Per-letter / phantom-token splits:** pypdfium renders some words across multiple chunks (`Sales` + `T` + `ransaction`, `JAKAR` + `TA` + `TA`, `PT.` + `T. BANK JASA`). A dedicated healer (`_heal_letter_splits` + `_dedupe_phantom_tokens`) deduplicates phantom tokens and rejoins single-letter splits so the description column reads cleanly downstream.

For exact column boundary calibration and gotchas per bank, see [architecture §8](../docs/architecture.md#8-per-bank-layout-reference).

---

## 8. Real-World Validation

### 8.1 BCA — `contoh_mutasi.pdf` (1 month, 11 pages)

| Metric | Result | Target |
|---|---:|---:|
| Transactions parsed | 134 | 134 (from the document's own summary) |
| Debits | 131 | 131 |
| Credits | 3 | 3 |
| Sum of debits | Rp 19,000,000.00 | Rp 19,000,000.00 *(exact)* |
| Sum of credits | Rp 21,000,000.00 | Rp 21,000,000.00 *(exact)* |
| Balance warnings | 0 | 0 |
| Parse warnings | 0 | 0 |

### 8.2 BRI — `mutasi_haswin/Mutasi_*.pdf` (12 months, 48 pages)

| Metric | Result |
|---|---:|
| Files processed | 12 |
| Transactions total | **951** (853 DB + 98 CR) |
| Balance warnings | **0** across the whole year |
| Parse warnings | **0** across the whole year |
| Per-file DB / CR sums | Match each PDF's own summary block exactly |

### 8.3 Mandiri — `sample_mandiri.pdf` (1 month, 2 pages)

| Metric | Result | Target |
|---|---:|---:|
| Bank auto-detected | `Mandiri` | `Mandiri` |
| Account name (reassembled from 2 chunks) | `BUDI SANTOSO` | same |
| Transactions parsed | 7 | 7 |
| Debits / Credits | 6 / 1 | 6 / 1 |
| Sum of credits (Dana Masuk) | Rp 5,000,000.00 | Rp 5,000,000.00 *(exact)* |
| Sum of debits (Dana Keluar) | Rp 674,000.00 | Rp 674,000.00 *(exact)* |
| Final saldo (Saldo Akhir) | Rp 5,326,000.00 | Rp 5,326,000.00 *(exact)* |
| Balance warnings | 0 | 0 |
| Parse warnings | 0 | 0 |

### 8.4 Cross-month classification (the prize)

Year-level totals returned by `/extract-batch`:

| Category | Count | Year sum (Rp) | Min single tx (Rp) | What it caught |
|---|---:|---:|---:|---|
| **Gaji**     | **12** | **120,000,000** |  9,500,000 | All 12 monthly `SAP-DD TRANSACTION` payroll deposits — recognised purely from cross-month recurrence (rule 5) |
| **THR**      | 1      |  23,000,000     | 23,000,000 | 1× `THR_Islam_2026` (rule 3) |
| **Bonus**    | 2      |  44,000,000     |  7,000,000 | 1× `BONUS_POOL_2025_1`, 1× `BONUS_INTERIM_2025` — all `BONUS_*` labels are Bonus (rule 1) |
| **Insentif** | 2      |  21,000,000     | 10,000,000 | 2× `ECUTI` extra-leave payouts (rule 2 — performance-tied) |
| Lainnya      | 81     |  50,000,000     |     10,000 | P2P transfers, refunds, etc. |

The same LLM call on per-month-isolated credits produced **0 Gaji detections**. Cross-month context turned 0 → 12 with 0.95 confidence — the most concrete validation possible that the batch endpoint solves a real problem.

---

## 9. Performance & Cost

Measured on Apple Silicon, Python 3.12.4, against the included samples.

| Operation | Time |
|---|---:|
| `extract_chunks` on a typical 2–4-page PDF | ~30 ms |
| `parse_transactions` on the chunks above | ~10 ms |
| Single `/extract` (BCA 11 pages, `classify=false`) | ~60 ms |
| Single `/extract` (Mandiri 2 pages, `classify=false`) | ~50 ms |
| Single `/extract` (BCA 11 pages, `classify=true`) | ~2.2 s *(LLM-dominated)* |
| `/extract-batch` over 3 PDFs (`classify=false`) | ~170 ms |
| `/extract-batch` over 12 BRI PDFs (`classify=true`) | **~29 s** *(one LLM call)* |
| Memory peak for 12-month batch | ~80 MB |

**Cost note:** the batch endpoint sends ~98 credit rows in one prompt (≤ a few KB of payload). At Azure `gpt-4.1-mini` pricing, a year of statements classifies for fractions of a cent — far cheaper than 12 individual calls in addition to being more accurate.

---

## 10. Project Layout

```
ocr_mutasi/                          ← project root
├── docs/
│   └── architecture.md              ← design rationale + per-bank reference
├── ocr_mutasi/                      ← Python package
│   ├── __init__.py                  ← version
│   ├── config.py                    ← .env loader (pydantic-settings)
│   ├── models.py                    ← typed Transaction / *Response models
│   ├── pdf_extractor.py             ← pypdfium2 → list[TextChunk]
│   │                                  raises InvalidPdfError
│   ├── parsers/                     ← per-bank table reconstruction
│   │   ├── __init__.py              ← detect_bank() + get_parser()
│   │   ├── common.py                ← shared helpers + generic Row
│   │   ├── bca.py                   ← BCA Rekening Tahapan
│   │   ├── bri.py                   ← BRI BritAma
│   │   ├── mandiri.py               ← Mandiri Tabungan e-Statement
│   │   ├── permata.py               ← Permata Rekening Koran
│   │   └── sinarmas.py              ← Sinarmas Tabungan
│   ├── llm_classifier.py            ← Azure OpenAI: single + batch
│   ├── pipeline.py                  ← run() + run_batch() + UnsupportedBankError
│   └── api.py                       ← FastAPI app
├── .env                             ← real secrets (gitignored)
├── .env.example                     ← committed template
├── .gitignore
├── README.md                        ← this file
├── requirements.txt
├── contoh_mutasi.pdf                ← BCA sample (1 month, 11 pages, 134 tx)
├── sample_mandiri.pdf               ← Mandiri sample (1 month, 2 pages, 7 tx)
└── mutasi_haswin/                   ← BRI samples (12 monthly statements)
```

---

## 11. Development Guide

### 11.1 Run the smoke tests

These import the package directly and don't touch the network. Use them after any parser change.

```bash
# BCA single-file
.venv/bin/python -c "
from pathlib import Path
from ocr_mutasi.pipeline import run
r = run(Path('contoh_mutasi.pdf').read_bytes(), classify=False)
assert r.audit.rows_detected == 134 and r.audit.credit_count == 3
assert not r.audit.balance_warnings and not r.audit.parse_warnings
print('BCA smoke: PASS')
"

# BRI batch (no LLM)
.venv/bin/python -c "
import glob
from pathlib import Path
from ocr_mutasi.pipeline import run_batch
files = sorted(glob.glob('mutasi_haswin/Mutasi_*.pdf'))
payload = [(p.split('/')[-1], Path(p).read_bytes()) for p in files]
r = run_batch(payload, classify=False)
assert r.audit.files_processed == 12 and r.audit.transactions_total == 951
assert all(not f.audit.balance_warnings and not f.audit.parse_warnings for f in r.files)
print('BRI batch smoke: PASS')
"

# Mandiri single-file
.venv/bin/python -c "
from pathlib import Path
from ocr_mutasi.pipeline import run
r = run(Path('sample_mandiri.pdf').read_bytes(), classify=False)
assert r.account.bank == 'Mandiri' and r.account.nama == 'BUDI SANTOSO'
assert r.audit.rows_detected == 7 and r.audit.debit_count == 6 and r.audit.credit_count == 1
assert not r.audit.balance_warnings and not r.audit.parse_warnings
print('Mandiri smoke: PASS')
"

# Three-bank mixed batch (no LLM) — confirms detect_bank + dispatch across all three layouts
.venv/bin/python -c "
from pathlib import Path
from ocr_mutasi.pipeline import run_batch
payload = [
    ('contoh_mutasi.pdf', Path('contoh_mutasi.pdf').read_bytes()),
    ('sample_mandiri.pdf', Path('sample_mandiri.pdf').read_bytes()),
    ('Mutasi_April_2026.pdf', Path('mutasi_haswin/Mutasi_April_2026.pdf').read_bytes()),
]
r = run_batch(payload, classify=False)
banks = sorted(f.account.bank for f in r.files)
assert banks == ['BCA', 'BRI', 'Mandiri']
print('mixed-bank batch:', {f.account.bank: f.audit.rows_detected for f in r.files})
"
```

### 11.2 End-to-end test against a live LLM

This makes a real Azure OpenAI call. Use sparingly.

```bash
.venv/bin/python -c "
import glob, time
from pathlib import Path
from ocr_mutasi.pipeline import run_batch
files = sorted(glob.glob('mutasi_haswin/Mutasi_*.pdf'))
payload = [(p.split('/')[-1], Path(p).read_bytes()) for p in files]
t0 = time.time()
r = run_batch(payload, classify=True)
print(f'{time.time()-t0:.1f}s — categories:', {k: (v.count, v.sum) for k, v in r.audit.category_totals.items()})
"
```

### 11.3 Add a new bank (high-level)

A new bank is a single new file plus two registration lines. See [architecture §17.1](../docs/architecture.md#171-adding-a-new-bank) for the detailed checklist. The short version:

1. Extract chunks from a sample PDF, observe column anchors and content positions.
2. Create `ocr_mutasi/parsers/<bank>.py` mirroring `bca.py` or `bri.py`:
   - Define a `_<bank>Layout` dataclass with one (x0, x1) per column.
   - Implement `_detect_column_layout`, `_column_boundaries`, `_chunks_to_rows`, `_group_into_blocks`, `_block_to_transaction`.
   - Expose `parse_header(chunks)` and `parse_transactions(chunks, header)`.
3. Register in `ocr_mutasi/parsers/__init__.py`:
   ```python
   from . import bca, bri, mybank
   # In detect_bank():
   if "<UNIQUE TITLE TOKEN>" in page1_text: return "MYBANK"
   # In get_parser():
   if bank == "MYBANK": return mybank
   ```
4. Validate with a smoke script. Confirm `audit.balance_warnings == []` against a real sample.

A bank parser typically takes 100–300 lines.

### 11.4 Inspect a PDF's raw chunks

When something looks off, the most useful debugging tool is dumping the chunks:

```bash
.venv/bin/python -c "
from ocr_mutasi.pdf_extractor import extract_chunks
for c in extract_chunks('mutasi_haswin/Mutasi_April_2026.pdf')[:40]:
    print(f'p{c.page} y={c.y0:7.2f} x={c.x0:7.2f}..{c.x1:7.2f} | {c.text!r}')
"
```

---

## 12. Configuration Reference

All settings are read from `.env` once at startup (singleton via `pydantic-settings`).

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | yes | — | Azure OpenAI resource URL, e.g. `https://my-resource.openai.azure.com/` |
| `AZURE_OPENAI_API_KEY` | yes | — | Resource key |
| `AZURE_OPENAI_API_VERSION` | yes | `2025-01-01-preview` | Azure API version |
| `AZURE_OPENAI_DEPLOYMENT` | yes | `gpt-4.1-mini` | Deployment name used for classification |
| `APP_HOST` | no | `0.0.0.0` | uvicorn bind address |
| `APP_PORT` | no | `8000` | uvicorn bind port |
| `LLM_REQUEST_TIMEOUT_S` | no | `30` | Per-request timeout in seconds (batch endpoint doubles this internally) |
| `MAX_PDF_BYTES` | no | `20000000` | Per-file upload cap. Exceeding → `413`. |

---

## 13. Troubleshooting

| Symptom | HTTP | Likely cause | What to do |
|---|---|---|---|
| `Could not read PDF: Failed to load document (PDFium: Data format error)` | 422 | Truncated, corrupt, or not a PDF | Confirm the file opens in a PDF viewer; re-upload |
| `PDF doesn't match any known bank layout` | 422 | Not BCA Rekening Tahapan, BRI BritAma, Mandiri Tabungan e-Statement, Permata Rekening Koran, or Sinarmas Tabungan | Confirm the bank/product on page 1. If it's a new bank, add a parser (see §11.3 / [architecture §17.1](../docs/architecture.md#171-adding-a-new-bank)). |
| `No transactions detected — is this a supported bank PDF?` | 422 | Header detected but body empty (e.g. statement is a cover page only) | Inspect the PDF; use §11.4 to dump raw chunks. |
| `account.nama` is `null` (but other fields populated) | 200 | The bank's header anchor (e.g. BCA `KCP <branch>`) doesn't match a variant on your statement | Run §11.4 against the upper-left and share the chunk dump; we'll extend the prefix list. |
| Mandiri amount comes out 1000× too small or as a date | 200 | Mandiri uses `.` thousands and `,` decimals (inverted vs BCA/BRI). If this regresses, the `_parse_id_amount` helper in `parsers/mandiri.py` is broken — bisect from there. | — |
| Empty `transactions` array with no error | 200 | Statement covers a period with no activity | Not an error; the audit block reports `rows_detected: 0` |
| Non-empty `audit.balance_warnings` | 200 | Running balance doesn't match a printed `Saldo` on some row — usually a missed or duplicated transaction | Inspect the offending page (line in the warning includes `p<page> <date>`). May indicate a layout drift bug. |
| Non-empty `audit.parse_warnings` | 200 | A row matched a header heuristic but its amount didn't parse | Look at the warning; usually a data oddity in one row |
| Non-empty `audit.classifier_errors` | 200 | Azure OpenAI was unreachable / returned bad JSON | Extraction data is still valid. Retry the request once Azure is healthy. |
| All credits classified `Lainnya`, even payroll-looking ones | 200 | You hit `/extract` per month instead of `/extract-batch` | Switch to the batch endpoint — single-PDF can't see cross-month recurrence (see §1 and §5.2). |
| `audit.classifier_errors` shows "Request timed out" and every credit is `category: null` | 200 | The LLM call for a large batch (hundreds of credits) exceeded `LLM_REQUEST_TIMEOUT_S`. The pipeline returns extraction data with unclassified credits and an error entry. | Bump `LLM_REQUEST_TIMEOUT_S` in `.env` to 120 (or higher for very busy accounts — the batch endpoint internally doubles this value). Default is 120s. |
| Big monthly salary deposits (e.g. via BCA `SMEMFTS`) classified as Lainnya | 200 | Pre-v0.7 prompt only listed `GAJI`/`SAP-DD`/`KR OTOMATIS` as payroll keywords. v0.7 adds `SMEMFTS` (BCA SME Mass Funds Transfer Service — the primary corporate salary channel). | Pull the latest build. |
| Allowance-looking rows via `LLG-DEUTSCHE BANK` / `LLG …` (often paired with `KR OTOMATIS`) classified as Gaji | 200 | `LLG` is BI's bulk-clearing channel for **allowances**, distinct from the main salary channel. v0.7 routes it through rule 3 → **Insentif**. | Pull the latest build. |
| A category looks wrong (e.g. `BONUS_INTERIM` classified as Insentif) | 200 | The LLM may be inferring instead of following rules — but every classified row's `reason` field cites which rule fired. Check the reason. | If the cited rule number doesn't match what §1 documents, the prompt drifted; see `SYSTEM_PROMPT` / `BATCH_SYSTEM_PROMPT` in `ocr_mutasi/llm_classifier.py`. |
| A `TUNJANGAN` row ended up in `Lainnya` | 200 | Pre-v0.6 behaviour — generic tunjangan used to fall into Lainnya. v0.6 routes work-related `TUNJANGAN <kind>` to **Insentif** via rule 3. | Pull the latest build and restart uvicorn. |
| Swagger UI shows "Add string item" for `files` instead of a file picker | n/a | Stale build — pull v0.5 (or restart uvicorn). v0.5 patches the OpenAPI schema to emit `format: "binary"`. | If still broken after restart, see FAQ §14. |
| `GET /` returns 404 in the browser | n/a | Pre-v0.5 build; `/` now `307`s to `/upload`. | Restart uvicorn. |
| Server log shows full `Traceback` on a malformed PDF | n/a | Should NOT happen — we re-raise as `InvalidPdfError` and log a one-line WARNING. If you see a traceback for a client-fault, that's a regression — please file. | — |

---

## 14. FAQ

**Q. Why is the project named "ocr_mutasi" if it doesn't do OCR?**
History. The first design called for PaddleOCR; we then discovered the source PDFs are digital with clean text layers and pivoted to direct text extraction. The name stuck. See [architecture §1](../docs/architecture.md#1-overview).

**Q. Can I add another bank (BNI / CIMB / …)?**
Yes — Mandiri (v0.5), Permata (v0.9), and Sinarmas (v0.10) were each one-file changes. See §11.3 and [architecture §17](../docs/architecture.md#17-extending-the-system) for the pattern.

**Q. Do scanned PDFs work?**
Yes, via a fallback. Digital statements (with a text layer) go through the fast, deterministic bank parsers. When no known bank layout matches — typically a scan/photo with no text layer — the pipeline OCRs the PDF through the shared **PaddleOCR service** (`ocr_common.paddle_ocr`) and asks the LLM to extract the transaction rows. Such results carry `account.bank = "OCR"` and a `parse_warnings` note. Deterministic parsing is always preferred; OCR + LLM is best-effort.

**Q. Is debit data sent to the LLM?**
No. Only credit rows (transactions with `type == "CR"`) are sent for classification. Debits are extracted and returned in `transactions` but never leave the local pipeline.

**Q. Why is `category_totals.Gaji.sum` so different from what I expected?**
The LLM classifies based on description + cross-month recurrence (in the batch case). If a deposit looks like salary to a human but lacks recurrence in the dataset (e.g. you only uploaded a single month), it'll be marked `Lainnya`. Use `/extract-batch` with as many months as you have.

**Q. Why two separate API endpoints instead of one with an optional list?**
Different request semantics: single-PDF uses `file:` (multipart, one field), batch uses `files:` (multipart array). Splitting them keeps the OpenAPI schema and client code clean.

**Q. How do I see exactly what prompt the LLM gets?**
Read `ocr_mutasi/llm_classifier.py` — `SYSTEM_PROMPT` (single) and `BATCH_SYSTEM_PROMPT` (cross-month) are at the top of the file, easy to tweak.

**Q. Can I run this without Azure OpenAI?**
Pass `?classify=false` on either endpoint. Extraction still runs; `credits` come back with `category: null`. Useful for debugging the parser without spending LLM budget.

**Q. Why is `TUNJANGAN TRANSPORT` (or `MAKAN`, `PULSA`, `KELUAR KOTA`, …) classified as Insentif instead of Lainnya?**
Because in Indonesian corporate-payroll convention, these are work-tied perks earned by an employee — closer to performance pay than to "generic other income". v0.6 routes any `TUNJANGAN <kind>` (except `TUNJANGAN HARI RAYA`, which goes to THR) to Insentif via rule 3 of the classifier prompt. See §1 for the full rule list, or the system-prompt source in `ocr_mutasi/llm_classifier.py`.

**Q. Why is `BONUS_INTERIM` classified as Bonus, not Insentif (since "interim" sounds performance-y)?**
The `BONUS_` prefix is the company's own bonus-program naming. Whether annual (`BONUS_POOL`) or mid-year (`BONUS_INTERIM`), every `BONUS_*` row goes to Bonus via rule 1. The distinction between annual and interim is internal to the bonus program, not a separate category.

**Q. How do I know which rule fired for a given classification?**
Every classified credit's `reason` field cites the rule (e.g. *"BONUS_INTERIM label → Bonus per rule 1"*, *"TUNJANGAN TRANSPORT → Insentif per rule 3"*). On the `/upload` page the reason shows in the rightmost column of each accordion table.

**Q. Why does the API emit OpenAPI 3.0.3 instead of 3.1?**
Because Swagger UI bundled with FastAPI doesn't fully implement OpenAPI 3.1's `contentMediaType` keyword for multi-file uploads — it falls back to rendering each file slot as an "Add string item" plain-text field. OpenAPI 3.0.3 uses the older `format: "binary"` keyword, which Swagger UI renders as a proper file picker. The downgrade is set on one line in `api.py`: `app.openapi_version = "3.0.3"`, plus a small `_custom_openapi` post-processor that rewrites any leftover `contentMediaType` keywords into `format: "binary"`.

**Q. Why is there a separate `/upload` page instead of just using Swagger UI?**
Because Swagger UI **cannot** render a single multi-file input (`<input type="file" multiple>`) — its array editor always renders one file slot per array item with an "Add item" button. No OpenAPI schema, in 3.0 or 3.1, can hint otherwise. The `/upload` page is plain HTML with a native multi-file input, so one click in the OS file picker uploads all selected PDFs in one request.

---

## 15. Limitations

- **Supported banks:** BCA Rekening Tahapan, BRI BritAma, Mandiri Tabungan e-Statement, Permata Rekening Koran, and Sinarmas Tabungan. Other layouts return `422`.
- **Digital PDFs only:** scanned PDFs (no text layer) return `422`. No OCR fallback in v1.
- **Use `/extract-batch` for a year of statements:** single-PDF classification cannot see cross-month recurrence — the dominant Gaji signal.
- **No persistence:** results are returned in the response and not stored.
- **No auth, no rate limiting:** deploy behind an internal gateway.
- **LLM determinism:** even at `temperature=0`, the API may produce slightly different reasoning text on identical inputs. Categories and confidence scores are stable on the validation data.

---

## 16. Roadmap

See [architecture §18](../docs/architecture.md#18-open-questions--future-work) for the design-side roadmap. The short list, in priority order:

1. **Auth & rate limiting** when this leaves the internal network.
2. **More banks** — BNI, CIMB, Danamon (each is a one-file change, see §11.3).
3. **Confidence-based human review** — surface low-confidence classifications in the UI.
4. **Multi-account merging** — if a user has BCA + BRI + Mandiri accounts, cross-bank recurrence is an even stronger Gaji signal.
5. **Scanned-PDF fallback** via PaddleOCR (opt-in).
6. **Persistence / job queue** for very long histories.

---

## 17. Data Privacy & Samples

This project parses real Indonesian bank statements. Those PDFs contain personal financial data — account numbers, names, addresses, every transaction, salary deposits — and must never land in a public (or even a "private" GitHub) repository.

### What's excluded from version control

The shipped `.gitignore` blocks:

- **All `*.pdf` files** by default. This is a blanket rule so accidental `git add` of a real statement is impossible.
- The `mutasi_haswin/` folder (BRI monthly statements) — ignored independently so even a typo-ed `git add mutasi_haswin/` does nothing.
- `.env` (carries the Azure OpenAI API key). Only `.env.example` is committed.
- Per-developer IDE state: `.vscode/`, `.idea/`, `.claude/`, `.agents/`, `.qodo/`, `skills-lock.json`.
- Virtualenvs (`.venv/`, `venv/`, `env/`), Python build artifacts, test caches, OS junk (`.DS_Store`, `Thumbs.db`).

### If you need to ship a sample PDF

Create a **synthetic** anonymised PDF (fake name, fake account number, fictional transactions) and override the rule explicitly with a `!` exception in `.gitignore`:

```gitignore
# .gitignore
*.pdf
!samples/synthetic_bca.pdf
!samples/synthetic_bri.pdf
!samples/synthetic_mandiri.pdf
```

### Pre-push safety checklist

Before the first `git push`:

```bash
# 1. Verify .gitignore is honoured — these should ALL show as ignored:
git check-ignore -v .env mutasi_haswin/Mutasi_April_2026.pdf sample_mandiri.pdf contoh_mutasi.pdf

# 2. Confirm nothing sensitive is staged for the first commit:
git status --short                         # PDFs / .env must NOT appear
git ls-files --others --exclude-standard | grep -E '\.(pdf|env)$' \
   && echo "⚠️  STOP — sensitive file would be added" \
   || echo "✓ clean"

# 3. Once pushed, rotate the Azure OpenAI key anyway (cheap insurance —
#    GitHub Secret Scanning is good but not infallible).
```

### What runs on the LLM

Only **credit rows** (`type == "CR"`) are sent to Azure OpenAI for classification. Debits stay local. See [architecture §14](../docs/architecture.md#14-security--pii) for the full data-flow review.
