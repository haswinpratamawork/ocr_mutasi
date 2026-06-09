#!/usr/bin/env python3
"""FastAPI wrapper for the salary slip parser.

One API request represents one nasabah/customer. Upload one or more PDFs for
that person, and the parser writes one combined extracted JSON and summary JSON.

The ``/parse`` response embeds the new Indonesian-keyed ``summary`` dict as well
as a backwards-compatible ``documents`` / ``totals`` projection so the existing
``/upload`` browser UI and the ocr_match orchestrator both keep working without
changes.
"""

from __future__ import annotations

import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from .extract_salary import parse_upload

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"


app = FastAPI(
    title="Salary Slip OCR Service API",
    description="Upload one or more salary-slip PDFs for one nasabah/customer per request.",
    version="1.0.0",
)


def patch_upload_file_schema(value: object) -> None:
    if isinstance(value, dict):
        if value.get("contentMediaType") == "application/octet-stream":
            value.pop("contentMediaType", None)
            value["format"] = "binary"
        for child in value.values():
            patch_upload_file_schema(child)
    elif isinstance(value, list):
        for child in value:
            patch_upload_file_schema(child)


def custom_openapi() -> dict:
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    patch_upload_file_schema(openapi_schema)
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]


async def save_uploads(files: list[UploadFile], upload_dir: Path) -> list[str]:
    saved_files = []
    upload_dir.mkdir(parents=True, exist_ok=True)

    for upload in files:
        file_name = Path(upload.filename or "").name
        if not file_name:
            continue
        if not file_name.casefold().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file_name} is not a PDF file.")

        target = upload_dir / file_name
        with target.open("wb") as output:
            shutil.copyfileobj(upload.file, output)
        saved_files.append(file_name)

    if not saved_files:
        raise HTTPException(status_code=400, detail="Upload at least one PDF file.")
    return saved_files


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "parser_folder": str(PARSER_DIR)}


@app.post("/parse")
async def parse_salary_slips(
    files: list[UploadFile] = File(..., description="One or more salary-slip PDFs for one nasabah."),
    password: Optional[str] = Form(None, description="Optional PDF password if the uploaded PDFs are protected."),
) -> JSONResponse:
    run_id = new_run_id()
    run_dir = RUNS_DIR / run_id
    upload_dir = run_dir / "uploads"
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = await save_uploads(files, upload_dir)
    summary = parse_upload(
        upload_dir,
        output_dir,
        password=password,
        allow_password_prompt=False,
    )
    needs_password = any(
        "password" in (error.get("kesalahan") or "").casefold()
        for error in summary.get("kesalahan", [])
    )
    # Re-emit the per-document and aggregate-totals views with English keys —
    # the existing /upload HTML and the ocr_match orchestrator both consume
    # this shape unchanged. The new Indonesian-keyed ``summary`` is still
    # available alongside for clients that want richer data.
    documents = [_document_compat(d) for d in summary.get("dokumen", [])]
    totals = _totals_compat(summary.get("periode", {}).get("total"))
    errors = [
        {"source_file": e.get("sumber_file") or e.get("source_file"),
         "error":       e.get("kesalahan")   or e.get("error")}
        for e in summary.get("kesalahan", [])
    ]
    return JSONResponse(
        {
            "ok": True,
            "needs_password": needs_password,
            "run_id": run_id,
            "uploaded_files": saved_files,
            # Compat keys for the /upload UI and ocr_match — same shape the
            # pre-rework API exposed.
            "documents": documents,
            "totals": totals,
            "document_count": len(documents),
            "errors": errors,
            # New Indonesian-keyed view (rework upstream's preferred shape).
            "summary": summary,
            "output_files": {
                "output_folder": str(output_dir),
                "extracted": str(output_dir / "extracted.json"),
                "summary": str(output_dir / "summary.json"),
            },
        }
    )


# --------------------------- compat shim ----------------------------------
# Map the new Indonesian-keyed per-document dict back to the English-keyed
# shape both the /upload HTML and the ocr_match service consume. Adds a
# safety-net for ``period`` (already YYYY-MM via tanggal_periode but may be a
# free-form Bulan Tahun string on some slips).

def _document_compat(d: dict[str, Any]) -> dict[str, Any]:
    """Project one ``summary.dokumen[*]`` entry to the legacy English keys.

    Note: the rework collapses ``tax`` and ``other_deduction`` into a single
    ``potongan`` (deduction) field — ``extract_llm.py`` documents this as
    *"deduction = all decreases combined into one number, including PPh/
    pajak/tax and any other deduction"*. The old vendored shape had separate
    tax and other_deduction fields; we no longer emit them.
    """
    return {
        "source_file":        d.get("sumber_file"),
        "worker_name":        d.get("nama_pekerja"),
        "institution_name":   d.get("nama_institusi"),
        "total_paid":         _num(d.get("total_dibayar")),
        "pokok":              _num(d.get("gaji_pokok")) or 0,
        "incentive":          _num(d.get("tunjangan")) or 0,
        "deduction":          _num(d.get("potongan")) or 0,
        "period":             d.get("tanggal_periode"),
        "extraction_method":  d.get("metode_ekstraksi") or "",
        "confidence_notes":   list(d.get("catatan") or []),
    }


def _totals_compat(totals_id: dict[str, Any] | None) -> dict[str, Any]:
    """Project ``summary.periode.total`` back to the legacy English keys.

    Same caveat as ``_document_compat``: aggregate ``tax`` / ``other_deduction``
    were dropped in the rework; the rework's single ``potongan`` total now
    encompasses both.
    """
    t = totals_id or {}
    return {
        "total_paid":      _num(t.get("total_dibayar")) or 0,
        "pokok":           _num(t.get("gaji_pokok")) or 0,
        "incentive":       _num(t.get("tunjangan")) or 0,
        "deduction":       _num(t.get("potongan")) or 0,
    }


def _num(value: Any) -> int | float | None:
    """Pass through numeric values, coerce nulls / empty strings to ``None``."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------- /upload browser page -------------------------
# Self-contained HTML drag-and-drop page reused verbatim from the pre-rework
# api_app.py — same UX (native multi-select via <input type="file" multiple>,
# per-document salary cards, aggregate totals card, raw-JSON panel). The
# response shape it depends on (``documents`` + ``totals``) is provided by
# the compat shim above.

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
    <p class="tagline">Drop one or more salary-slip PDFs and get back the parsed worker, institution, take-home pay, and a pokok / incentive / deduction breakdown. Multi-select with <kbd>Cmd</kbd>/<kbd>Ctrl</kbd>-click. &nbsp;·&nbsp; <a href="/docs">Swagger UI</a> &nbsp;·&nbsp; <a href="/redoc">ReDoc</a></p>

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
    const r = await fetch(`/parse`,
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
    ['',          'Deduction (sum)', t.deduction],
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
            <tr><td>Deduction <span class="hint">(incl. tax)</span></td><td>${fmtRp(d.deduction)}</td></tr>
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


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/upload", status_code=307)


@app.get("/upload", response_class=HTMLResponse, include_in_schema=False)
def upload_page() -> HTMLResponse:
    return HTMLResponse(_UPLOAD_PAGE)
