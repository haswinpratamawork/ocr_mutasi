#!/usr/bin/env python3
"""FastAPI service for salary-slip parsing."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# pikepdf is used as a pre-processor for password-protected PDFs: we open
# the encrypted PDF with the caller-supplied password and write an
# unencrypted copy to a temp file that the existing salary_slip_parser
# can read without further changes. Owner-only locks are stripped legitimately;
# user passwords are never brute-forced — if the supplied password is wrong
# we return a clear 422.
import pikepdf

from salary_slip_parser import (
    AutoOcrPdfTextExtractor,
    ParserConfig,
    SalarySlipAnalyzer,
    summary_to_jsonable,
)


BASE_SALARY_KEYWORDS = (
    "gaji pokok",
    "upah pokok",
    "basic salary",
    "base salary",
    "base compensation",
    "imbalan dasar",
    "pokok",
)


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])


class ApiInfoResponse(BaseModel):
    service: str
    docs_url: str
    parse_endpoint: str
    health_endpoint: str


class ParsedDocument(BaseModel):
    source_file: str
    worker_name: str | None
    institution_name: str | None
    total_paid: int | float | None
    pokok: int | float
    tax: int | float
    incentive: int | float
    deduction: int | float
    other_deduction: int | float
    period: str | None = None  # YYYY-MM, extracted from the slip's "Period:" line
    confidence_notes: list[str]
    extraction_method: str


class ParseError(BaseModel):
    source_file: str
    error: str


class ParseTotals(BaseModel):
    total_paid: int | float
    pokok: int | float
    tax: int | float
    incentive: int | float
    deduction: int | float
    other_deduction: int | float


class ParseResponse(BaseModel):
    generated_at: str
    document_count: int
    totals: ParseTotals
    documents: list[ParsedDocument]
    errors: list[ParseError]

app = FastAPI(
    title="Salary Slip Parser API",
    version="1.0.0",
    description="Local API for extracting compact salary-slip JSON.",
    openapi_version="3.0.3",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _binary_file_schema(schema_part):
    if isinstance(schema_part, dict):
        if schema_part.get("contentMediaType") == "application/octet-stream":
            schema_part.pop("contentMediaType", None)
            schema_part["format"] = "binary"
        for value in schema_part.values():
            _binary_file_schema(value)
    elif isinstance(schema_part, list):
        for item in schema_part:
            _binary_file_schema(item)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    _binary_file_schema(schema)
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.get("/", response_model=ApiInfoResponse)
def api_info() -> dict:
    return {
        "service": "Salary Slip Parser API",
        "docs_url": "/docs",
        "parse_endpoint": "POST /parse",
        "health_endpoint": "GET /health",
    }


def is_pdf(filename: str) -> bool:
    return Path(filename).suffix.lower() == ".pdf"


def base_salary_total(summary: dict) -> int | float:
    total = 0
    for item in summary.get("earnings", []):
        label = item.get("label", "").casefold()
        if any(keyword in label for keyword in BASE_SALARY_KEYWORDS):
            total += item.get("amount") or 0
    return total


def compact_result(summary: dict) -> dict:
    pokok = base_salary_total(summary)
    return {
        "source_file": Path(summary["source_file"]).name,
        "worker_name": summary.get("worker_name"),
        "institution_name": summary.get("institution"),
        "total_paid": summary.get("paid_salary_total"),
        "pokok": pokok,
        "tax": summary.get("tax_cutoff_total") or 0,
        "incentive": summary.get("incentive_total") or 0,
        "deduction": summary.get("deduction_total") or 0,
        "other_deduction": summary.get("other_cutoff_total") or 0,
        "period": summary.get("period"),
        "confidence_notes": summary.get("confidence_notes", []),
    }


class PdfPasswordRequiredError(Exception):
    """The PDF is encrypted and the supplied password (if any) didn't open it."""


def _decrypt_if_needed(pdf_path: Path, password: str | None) -> Path:
    """If ``pdf_path`` is password-protected, decrypt it using ``password`` and
    write an unencrypted copy alongside it. Returns the path the rest of the
    pipeline should read.

    Strategy:
      1. If pikepdf can open the file with no password, it's not encrypted —
         return the original path.
      2. Otherwise try the caller's password; on success, save an
         unencrypted copy and return that path.
      3. On password failure, raise PdfPasswordRequiredError.
    """
    # Cheap probe: try empty password. pikepdf opens unencrypted PDFs fine
    # this way too.
    try:
        with pikepdf.open(pdf_path) as _:
            return pdf_path
    except pikepdf.PasswordError:
        pass  # really is encrypted; fall through
    except pikepdf.PdfError:
        return pdf_path  # malformed — let the parser fail with its own error
    # Need the caller's password.
    try:
        with pikepdf.open(pdf_path, password=password or "") as pdf:
            decrypted_path = pdf_path.with_suffix(".dec.pdf")
            pdf.save(decrypted_path)  # save() drops encryption by default
            return decrypted_path
    except pikepdf.PasswordError as exc:
        raise PdfPasswordRequiredError(
            "PDF is password-protected and the supplied password (if any) is incorrect"
        ) from exc


def parse_pdf(pdf_path: Path, original_name: str, ocr: str, password: str | None = None) -> dict:
    config = ParserConfig()
    extractor = AutoOcrPdfTextExtractor(config=config, ocr_mode=ocr)
    analyzer = SalarySlipAnalyzer(config)

    readable_path = _decrypt_if_needed(pdf_path, password)
    extracted = extractor.extract(readable_path)
    extracted["source_file"] = original_name
    summary = summary_to_jsonable(analyzer.analyze(extracted))
    summary["source_file"] = original_name
    result = compact_result(summary)
    result["extraction_method"] = extracted.get("extraction_method", "pdf_text")
    return result


@app.get("/health", response_model=HealthResponse)
def health() -> dict:
    return {"status": "ok"}


@app.post("/parse", response_model=ParseResponse)
def parse_salary_slips(
    files: Annotated[list[UploadFile], File(description="One or more PDF salary slips.")],
    ocr: str = "auto",
    password: Annotated[
        str | None,
        Form(description="Optional PDF password applied to every file in the batch."),
    ] = None,
) -> dict:
    if ocr not in {"auto", "never", "always"}:
        raise HTTPException(status_code=400, detail="ocr must be one of: auto, never, always")
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one PDF file.")

    documents = []
    errors = []

    with TemporaryDirectory(prefix="salary-slip-api-") as temp_dir:
        temp_path = Path(temp_dir)

        for upload in files:
            original_name = upload.filename or "unnamed.pdf"
            if not is_pdf(original_name):
                errors.append({"source_file": original_name, "error": "Only PDF files are supported."})
                continue

            pdf_path = temp_path / Path(original_name).name
            with pdf_path.open("wb") as output:
                shutil.copyfileobj(upload.file, output)

            try:
                documents.append(parse_pdf(pdf_path, original_name, ocr, password=password))
            except PdfPasswordRequiredError as exc:
                errors.append({
                    "source_file": original_name,
                    "error": (
                        "PDF is password-protected. Pass the password as the "
                        "'password' form field."
                    ),
                })
            except Exception as exc:
                errors.append({"source_file": original_name, "error": str(exc)})

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "document_count": len(documents),
        "totals": {
            "total_paid": sum(item["total_paid"] or 0 for item in documents),
            "pokok": sum(item["pokok"] or 0 for item in documents),
            "tax": sum(item["tax"] or 0 for item in documents),
            "incentive": sum(item["incentive"] or 0 for item in documents),
            "deduction": sum(item["deduction"] or 0 for item in documents),
            "other_deduction": sum(item["other_deduction"] or 0 for item in documents),
        },
        "documents": documents,
        "errors": errors,
    }


@app.post("/parse-one", response_model=ParsedDocument)
def parse_one_salary_slip(
    file: Annotated[UploadFile, File(description="One PDF salary slip.")],
    ocr: str = "auto",
    password: Annotated[
        str | None,
        Form(description="Optional PDF password if the file is encrypted."),
    ] = None,
) -> dict:
    if ocr not in {"auto", "never", "always"}:
        raise HTTPException(status_code=400, detail="ocr must be one of: auto, never, always")

    original_name = file.filename or "unnamed.pdf"
    if not is_pdf(original_name):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with TemporaryDirectory(prefix="salary-slip-api-") as temp_dir:
        pdf_path = Path(temp_dir) / Path(original_name).name
        with pdf_path.open("wb") as output:
            shutil.copyfileobj(file.file, output)
        try:
            return parse_pdf(pdf_path, original_name, ocr, password=password)
        except PdfPasswordRequiredError as exc:
            raise HTTPException(
                status_code=422,
                detail=(
                    "PDF is password-protected. Pass the password as the 'password' "
                    "form field."
                ),
            ) from exc


# ---------------------------------------------------------------------------
# /upload — self-contained HTML drag-and-drop page
#
# This route exists for the same reason as the matching one on the
# ocr_mutasi service: Swagger UI insists on rendering an "Add string item"
# button per file slot, which is awkward for multi-file PDF uploads. The
# native <input type="file" multiple> below lets the OS file picker handle
# multi-selection (Cmd-click on macOS, Ctrl-click on Windows/Linux) in a
# single click. After upload, the page renders per-document salary cards
# (worker / institution / take-home pay / pokok-tax-incentive-deduction
# breakdown), an aggregate totals card across all files, and a collapsible
# raw-JSON panel.
# ---------------------------------------------------------------------------

_UPLOAD_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Salary Slip Parser — Upload</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root { --fg: #1a1a1a; --muted: #6b7280; --bg: #f7f7f8; --panel: #fff;
            --border: #e5e7eb; --accent: #2563eb; --accent-hover: #1d4ed8;
            --ok: #059669; --err: #dc2626; --warn: #f59e0b;
            --code-bg: #0f172a; --code-fg: #e2e8f0; }
    * { box-sizing: border-box; }
    body { margin: 0; padding: 24px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           background: var(--bg); color: var(--fg); }
    .container { max-width: 980px; margin: 0 auto; }
    h1 { margin: 0 0 4px; font-size: 22px; }
    p.tagline { margin: 0 0 24px; color: var(--muted); font-size: 14px; }
    .card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
            padding: 20px; margin-bottom: 16px; }
    label.file-drop { display: block; border: 2px dashed var(--border); border-radius: 8px;
                      padding: 32px 16px; text-align: center; cursor: pointer;
                      transition: border-color .15s, background .15s; }
    label.file-drop:hover { border-color: var(--accent); background: #f0f7ff; }
    label.file-drop.has-files { border-color: var(--ok); background: #f0fdf4; }
    label.file-drop .icon { font-size: 32px; line-height: 1; }
    label.file-drop .primary { display: block; margin-top: 8px; font-weight: 600; }
    label.file-drop .secondary { display: block; margin-top: 4px; color: var(--muted); font-size: 13px; }
    input[type=file] { display: none; }
    #file-list { margin-top: 12px; padding: 0; list-style: none; font-size: 13px; }
    #file-list li { padding: 4px 0; color: var(--muted); }
    .pw-row { margin-top: 14px; }
    .pw-row label { display: block; font-size: 12px; text-transform: uppercase;
                    letter-spacing: .04em; color: var(--muted); margin-bottom: 4px; }
    .pw-row input[type=password] { width: 100%; padding: 8px 10px; border-radius: 6px;
                                   border: 1px solid var(--border); font: inherit; font-size: 13px; }
    .pw-row .hint { font-size: 12px; color: var(--muted); margin-top: 4px; }
    .controls { display: flex; gap: 16px; align-items: center; margin-top: 16px; flex-wrap: wrap; }
    button { font: inherit; font-weight: 600; padding: 10px 18px; border-radius: 6px;
             border: none; background: var(--accent); color: #fff; cursor: pointer; }
    button:hover { background: var(--accent-hover); }
    button:disabled { background: #94a3b8; cursor: not-allowed; }
    label.opt { display: flex; gap: 6px; align-items: center; font-size: 14px; color: var(--muted); }
    label.opt select { font: inherit; padding: 4px 8px; border-radius: 4px; border: 1px solid var(--border); }
    #status { margin: 8px 0; font-size: 13px; min-height: 18px; }
    #status.ok { color: var(--ok); }
    #status.err { color: var(--err); }
    /* Aggregate totals card */
    #totals { margin-top: 16px; }
    #totals h2 { margin: 0 0 12px; font-size: 16px; }
    #totals .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                    gap: 8px; }
    #totals .grid .cell { padding: 10px 12px; background: #fafafb; border-radius: 6px;
                          border: 1px solid var(--border); }
    #totals .grid .cell .label { display: block; font-size: 11px; text-transform: uppercase;
                                 letter-spacing: .04em; color: var(--muted); margin-bottom: 4px; }
    #totals .grid .cell .value { font-size: 15px; font-weight: 600; font-variant-numeric: tabular-nums; }
    #totals .grid .cell.take-home { background: #f0fdf4; border-color: #86efac; }
    #totals .grid .cell.take-home .value { color: var(--ok); font-size: 17px; }
    /* Per-document cards */
    #docs .doc { background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
                 padding: 16px 18px; margin-bottom: 12px; }
    #docs .doc h3 { margin: 0 0 4px; font-size: 16px; }
    #docs .doc .meta { color: var(--muted); font-size: 13px; margin-bottom: 12px; }
    #docs .doc .meta .src { font-family: ui-monospace, "SF Mono", Menlo, monospace;
                            font-size: 12px; color: var(--accent); }
    #docs .doc .take-home { background: #f0fdf4; border-radius: 6px; padding: 10px 14px;
                            margin-bottom: 12px; display: flex; align-items: baseline; gap: 12px; }
    #docs .doc .take-home .label { font-size: 12px; text-transform: uppercase;
                                   letter-spacing: .04em; color: var(--muted); }
    #docs .doc .take-home .value { font-size: 20px; font-weight: 700; color: var(--ok);
                                   font-variant-numeric: tabular-nums; }
    #docs .doc table.breakdown { width: 100%; border-collapse: collapse; font-size: 13px;
                                 font-variant-numeric: tabular-nums; }
    #docs .doc table.breakdown th { text-align: left; font-weight: 600; color: var(--muted);
                                    font-size: 11px; text-transform: uppercase;
                                    letter-spacing: .04em; padding: 6px 8px;
                                    border-bottom: 1px solid var(--border); }
    #docs .doc table.breakdown td { padding: 8px; text-align: right; }
    #docs .doc table.breakdown td:first-child { text-align: left; font-weight: 500; }
    #docs .doc .notes { margin-top: 10px; padding: 8px 10px; background: #fffbeb;
                        border-left: 3px solid var(--warn); font-size: 12px; color: #92400e; }
    #docs .doc .notes ul { margin: 4px 0 0; padding-left: 16px; }
    /* Errors panel */
    #errors { background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px;
              padding: 14px 18px; margin-top: 12px; }
    #errors h2 { margin: 0 0 8px; font-size: 14px; color: var(--err); }
    #errors li { font-size: 13px; color: var(--err); }
    /* Raw-JSON toggle */
    details { margin-top: 12px; }
    summary { cursor: pointer; color: var(--muted); font-size: 13px; }
    pre#response { background: var(--code-bg); color: var(--code-fg); padding: 16px; border-radius: 6px;
                   overflow: auto; font-size: 12px; line-height: 1.5; max-height: 480px; margin: 0; }
    a { color: var(--accent); }
    code { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px;
           background: #f3f4f6; padding: 1px 5px; border-radius: 3px; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Salary Slip Parser — Upload</h1>
    <p class="tagline">Drop one or more salary-slip PDFs and get back the parsed worker, institution, take-home pay, and a pokok / tax / incentive / deduction breakdown. Multi-select with <kbd>Cmd</kbd>/<kbd>Ctrl</kbd>-click. &nbsp;·&nbsp; <a href="/docs">Swagger UI</a> &nbsp;·&nbsp; <a href="/redoc">ReDoc</a></p>

    <form id="form" class="card">
      <label for="files" class="file-drop" id="drop">
        <span class="icon">🧾</span>
        <span class="primary">Click to choose salary-slip PDFs</span>
        <span class="secondary">Multi-select supported · accepts <code>.pdf</code></span>
        <input type="file" id="files" name="files" accept="application/pdf,.pdf" multiple>
      </label>
      <ul id="file-list"></ul>

      <div class="pw-row">
        <label for="password">PDF password (optional)</label>
        <input type="password" id="password" name="password" placeholder="Leave blank for unencrypted PDFs">
        <div class="hint">Salary slips are sometimes locked with employee ID, NIK, or birthdate. The same value is applied to every file in this upload.</div>
      </div>

      <div class="controls">
        <label class="opt">OCR fallback:
          <select id="ocr">
            <option value="auto" selected>auto (use OCR only if no text layer)</option>
            <option value="never">never (PDF text only)</option>
            <option value="always">always (force Apple Vision OCR)</option>
          </select>
        </label>
        <button type="submit" id="go">Parse</button>
      </div>
      <div id="status"></div>
    </form>

    <div id="totals"></div>
    <div id="docs"></div>
    <div id="errors" style="display:none"></div>

    <details>
      <summary>Show raw JSON response</summary>
      <pre id="response">(no request sent yet)</pre>
    </details>
  </div>

<script>
const form     = document.getElementById('form');
const filesIn  = document.getElementById('files');
const drop     = document.getElementById('drop');
const list     = document.getElementById('file-list');
const status   = document.getElementById('status');
const totals   = document.getElementById('totals');
const docs     = document.getElementById('docs');
const errors   = document.getElementById('errors');
const responseEl = document.getElementById('response');
const goBtn    = document.getElementById('go');
const ocrSel   = document.getElementById('ocr');

function esc(s) {
  if (s == null) return '';
  return String(s).replace(/[&<>"']/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function fmtBytes(n) {
  if (n < 1024) return n + ' B';
  if (n < 1024*1024) return (n/1024).toFixed(1) + ' KB';
  return (n/1024/1024).toFixed(2) + ' MB';
}
function fmtRp(n) {
  if (n == null) return '—';
  return 'Rp ' + Number(n).toLocaleString('id-ID',
    { maximumFractionDigits: 2, minimumFractionDigits: 0 });
}

filesIn.addEventListener('change', () => {
  list.innerHTML = '';
  if (filesIn.files.length === 0) {
    drop.classList.remove('has-files');
    drop.querySelector('.primary').textContent = 'Click to choose salary-slip PDFs';
    return;
  }
  drop.classList.add('has-files');
  drop.querySelector('.primary').textContent =
    `${filesIn.files.length} file(s) selected — click to change`;
  for (const f of filesIn.files) {
    const li = document.createElement('li');
    li.textContent = `• ${f.name} (${fmtBytes(f.size)})`;
    list.appendChild(li);
  }
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (filesIn.files.length === 0) {
    status.className = 'err';
    status.textContent = 'Choose at least one PDF first.';
    return;
  }
  const fd = new FormData();
  for (const f of filesIn.files) fd.append('files', f, f.name);
  const pw = document.getElementById('password').value;
  if (pw) fd.append('password', pw);
  const ocr = ocrSel.value;
  status.className = '';
  status.textContent = `Parsing ${filesIn.files.length} file(s) (ocr=${ocr})…`;
  totals.innerHTML = '';
  docs.innerHTML = '';
  errors.style.display = 'none';
  errors.innerHTML = '';
  responseEl.textContent = '';
  goBtn.disabled = true;
  const t0 = performance.now();
  try {
    const r = await fetch(`/parse?ocr=${encodeURIComponent(ocr)}`,
                         { method: 'POST', body: fd });
    const data = await r.json();
    const dt = ((performance.now() - t0) / 1000).toFixed(2);
    responseEl.textContent = JSON.stringify(data, null, 2);
    if (!r.ok) {
      status.className = 'err';
      status.textContent = `HTTP ${r.status} in ${dt}s — ${data.detail || 'error'}`;
      return;
    }
    status.className = 'ok';
    status.textContent =
      `HTTP ${r.status} in ${dt}s — parsed ${data.document_count} document(s)`
      + (data.errors && data.errors.length ? `, ${data.errors.length} error(s)` : '');
    renderTotals(data);
    renderDocs(data);
    renderErrors(data);
  } catch (err) {
    status.className = 'err';
    status.textContent = `Network error: ${err}`;
  } finally {
    goBtn.disabled = false;
  }
});

function renderTotals(data) {
  if (data.document_count <= 1) return;  // single-file: totals == per-doc card, redundant
  const t = data.totals || {};
  const cells = [
    ['take-home', 'Take-home (sum)', t.total_paid],
    ['',          'Pokok (sum)',     t.pokok],
    ['',          'Incentive (sum)', t.incentive],
    ['',          'Tax (sum)',       t.tax],
    ['',          'Deduction (sum)', t.deduction],
    ['',          'Other (sum)',     t.other_deduction],
  ];
  totals.innerHTML = `
    <div class="card">
      <h2>Aggregate across ${data.document_count} slips</h2>
      <div class="grid">
        ${cells.map(([cls, label, value]) => `
          <div class="cell ${cls}">
            <span class="label">${esc(label)}</span>
            <span class="value">${fmtRp(value)}</span>
          </div>`).join('')}
      </div>
    </div>`;
}

function renderDocs(data) {
  const items = data.documents || [];
  if (items.length === 0) {
    docs.innerHTML = `<div class="card" style="text-align:center;color:var(--muted)">No documents parsed.</div>`;
    return;
  }
  docs.innerHTML = items.map(d => {
    const notes = (d.confidence_notes || []);
    return `
      <div class="doc">
        <h3>${esc(d.worker_name) || '(unknown worker)'}</h3>
        <div class="meta">
          ${esc(d.institution_name) || '(unknown institution)'}
          &nbsp;·&nbsp; <span class="src">${esc(d.source_file)}</span>
          &nbsp;·&nbsp; ${esc(d.extraction_method || 'pdf_text')}
        </div>
        <div class="take-home">
          <span class="label">Take-home pay</span>
          <span class="value">${fmtRp(d.total_paid)}</span>
        </div>
        <table class="breakdown">
          <thead><tr>
            <th>Component</th><th>Amount</th>
          </tr></thead>
          <tbody>
            <tr><td>Pokok (basic salary)</td><td>${fmtRp(d.pokok)}</td></tr>
            <tr><td>Incentive / tunjangan</td><td>${fmtRp(d.incentive)}</td></tr>
            <tr><td>Tax (PPh / withholding)</td><td>${fmtRp(d.tax)}</td></tr>
            <tr><td>Deduction</td><td>${fmtRp(d.deduction)}</td></tr>
            <tr><td>Other cut-offs</td><td>${fmtRp(d.other_deduction)}</td></tr>
          </tbody>
        </table>
        ${notes.length ? `
          <div class="notes">
            <strong>Confidence notes:</strong>
            <ul>${notes.map(n => `<li>${esc(n)}</li>`).join('')}</ul>
          </div>` : ''}
      </div>`;
  }).join('');
}

function renderErrors(data) {
  const errs = data.errors || [];
  if (errs.length === 0) return;
  errors.style.display = 'block';
  errors.innerHTML = `
    <h2>${errs.length} file(s) failed</h2>
    <ul>${errs.map(e => `
      <li><strong>${esc(e.source_file)}</strong>: ${esc(e.error)}</li>`).join('')}</ul>`;
}
</script>
</body>
</html>"""


@app.get("/upload", response_class=HTMLResponse, include_in_schema=False)
def upload_page() -> HTMLResponse:
    """Self-contained HTML drag-and-drop UI for the /parse endpoint.

    Mirrors the design of the matching page on the ocr_mutasi service so
    both APIs feel like part of the same toolkit. Uses a native
    `<input type="file" multiple>` so the OS file picker handles
    multi-selection — Swagger UI can't render that.
    """
    return HTMLResponse(_UPLOAD_PAGE)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    """Quiet the browser's auto-request so the access log stays clean."""
    return Response(status_code=204)
