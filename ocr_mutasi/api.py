"""FastAPI surface for the mutasi-extraction service.

Exposes a single business endpoint plus a health check. Keep this thin —
all logic lives in `pipeline.run`.

Run locally:  uvicorn ocr_mutasi.api:app --reload
"""
from __future__ import annotations

import logging
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse, RedirectResponse

from . import __version__
from .config import get_settings
from .models import BatchExtractionResponse, ExtractionResponse
from .pdf_extractor import InvalidPdfError, PdfPasswordRequiredError
from .pipeline import UnsupportedBankError, run as run_pipeline, run_batch as run_pipeline_batch

logger = logging.getLogger("ocr_mutasi.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="OCR Mutasi",
    version=__version__,
    description="Extract BCA/BRI/Mandiri bank-statement transactions from PDF and classify credits (Gaji / THR / Bonus / Insentif / Lainnya).",
)
# Emit OpenAPI 3.0.3 instead of FastAPI's default 3.1.0. Swagger UI bundled with
# FastAPI doesn't fully implement 3.1's `contentMediaType` keyword for multi-file
# uploads — it renders the file array as plain "Add string item" rows.
app.openapi_version = "3.0.3"


def _custom_openapi() -> dict:
    """Patch FastAPI's generated schema so multi-file uploads render as a
    real file picker in Swagger UI.

    Even when we ask for OpenAPI 3.0.3, FastAPI still emits binary fields
    using the 3.1-only `contentMediaType` keyword. That's invalid 3.0 and
    Swagger UI silently falls back to a plain-text "Add string item" widget.
    We walk the schema and rewrite every
        {"type": "string", "contentMediaType": "application/octet-stream"}
    into the 3.0-native
        {"type": "string", "format": "binary"}
    which Swagger UI does render as a file picker (with multi-select when
    the field's parent is an array).
    """
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
    )

    def rewrite(node: object) -> None:
        if isinstance(node, dict):
            if (node.get("type") == "string"
                    and node.get("contentMediaType") == "application/octet-stream"):
                node.pop("contentMediaType", None)
                node["format"] = "binary"
            for v in node.values():
                rewrite(v)
        elif isinstance(node, list):
            for item in node:
                rewrite(item)

    rewrite(schema)
    app.openapi_schema = schema
    return schema


app.openapi = _custom_openapi


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect the bare URL to the friendly upload page (which links onward
    to Swagger UI and ReDoc for API exploration)."""
    return RedirectResponse(url="/upload", status_code=307)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    """Swallow the browser's auto-request for a tab icon to avoid noisy 404s
    in the access log. We don't ship an icon; 204 No Content is the polite
    answer."""
    return Response(status_code=204)


_UPLOAD_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>OCR Mutasi — Upload</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root { --fg: #1a1a1a; --muted: #6b7280; --bg: #f7f7f8; --panel: #fff;
            --border: #e5e7eb; --accent: #2563eb; --accent-hover: #1d4ed8;
            --ok: #059669; --err: #dc2626; --code-bg: #0f172a; --code-fg: #e2e8f0; }
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
    label.opt { display: flex; gap: 6px; align-items: center; font-size: 14px; color: var(--muted); cursor: pointer; }
    #status { margin: 8px 0; font-size: 13px; min-height: 18px; }
    #status.ok { color: var(--ok); }
    #status.err { color: var(--err); }
    #summary { margin-top: 16px; }
    #summary .cat { border: 1px solid var(--border); border-radius: 8px;
                    margin-bottom: 10px; background: var(--panel); overflow: hidden; }
    #summary .cat > summary { list-style: none; cursor: pointer; padding: 14px 16px;
                              display: grid;
                              grid-template-columns: 24px 110px 1fr auto;
                              gap: 12px; align-items: center; user-select: none; }
    #summary .cat > summary::-webkit-details-marker { display: none; }
    #summary .cat > summary::before { content: "▸"; color: var(--muted); transition: transform .15s; display: inline-block; }
    #summary .cat[open] > summary::before { transform: rotate(90deg); }
    #summary .cat .cat-name { font-weight: 600; font-size: 15px; }
    #summary .cat .cat-meta { color: var(--muted); font-size: 13px; }
    #summary .cat .cat-meta .sep { margin: 0 8px; opacity: .5; }
    #summary .cat .cat-sum { font-variant-numeric: tabular-nums; font-weight: 600; }
    #summary .cat[data-category="Gaji"] .cat-name { color: var(--ok); }
    #summary .cat[data-category="THR"] .cat-name { color: #f59e0b; }
    #summary .cat[data-category="Bonus"] .cat-name { color: #a855f7; }
    #summary .cat[data-category="Insentif"] .cat-name { color: #06b6d4; }
    #summary .tx-wrap { overflow-x: auto; border-top: 1px solid var(--border); }
    #summary .tx-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    #summary .tx-table th, #summary .tx-table td { padding: 8px 12px; text-align: left;
                                                   border-bottom: 1px solid var(--border); vertical-align: top; }
    #summary .tx-table tr:last-child td { border-bottom: none; }
    #summary .tx-table th { background: #fafafb; font-weight: 600; color: var(--muted);
                            font-size: 12px; text-transform: uppercase; letter-spacing: .03em; }
    #summary .tx-table td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
    #summary .tx-table td.desc { max-width: 320px; word-break: break-word; }
    #summary .tx-table td.reason { max-width: 280px; color: var(--muted); }
    #summary .tx-table td.src { white-space: nowrap; color: var(--muted); font-size: 12px; }
    #summary .empty { padding: 16px; color: var(--muted); font-style: italic; text-align: center; }
    pre#response { background: var(--code-bg); color: var(--code-fg); padding: 16px; border-radius: 6px;
                   overflow: auto; font-size: 12px; line-height: 1.5; max-height: 480px; margin: 0; }
    details { margin-top: 12px; }
    summary { cursor: pointer; color: var(--muted); font-size: 13px; }
    a { color: var(--accent); }
  </style>
</head>
<body>
  <div class="container">
    <h1>OCR Mutasi — Batch upload</h1>
    <p class="tagline">Select one or more BCA / BRI mutation PDFs. Hold <kbd>Cmd</kbd> (macOS) or <kbd>Ctrl</kbd> (Windows / Linux) in the file picker to multi-select. All credits across all files are classified together in one LLM call. &nbsp;·&nbsp; <a href="/docs">Swagger UI</a> &nbsp;·&nbsp; <a href="/redoc">ReDoc</a></p>

    <form id="form" class="card">
      <label for="files" class="file-drop" id="drop">
        <span class="icon">📄</span>
        <span class="primary">Click to choose PDFs</span>
        <span class="secondary">Multi-select supported · accepts <code>.pdf</code></span>
        <input type="file" id="files" name="files" accept="application/pdf,.pdf" multiple>
      </label>
      <ul id="file-list"></ul>

      <div class="pw-row">
        <label for="password">PDF password (optional)</label>
        <input type="password" id="password" name="password" placeholder="Leave blank for unencrypted PDFs">
        <div class="hint">Indonesian e-statements commonly use the last 6 digits of the account number, the birthdate in DDMMYYYY, or NIK. The same value is applied to every file in this upload.</div>
      </div>

      <div class="controls">
        <label class="opt"><input type="checkbox" id="classify" checked> Classify credits with LLM (Gaji / THR / Bonus / Insentif)</label>
        <button type="submit" id="go">Extract</button>
      </div>
      <div id="status"></div>
    </form>

    <div id="summary"></div>

    <details>
      <summary>Show raw JSON response</summary>
      <pre id="response">(no request sent yet)</pre>
    </details>
  </div>

<script>
const form = document.getElementById('form');
const filesIn = document.getElementById('files');
const drop = document.getElementById('drop');
const list = document.getElementById('file-list');
const status = document.getElementById('status');
const summary = document.getElementById('summary');
const responseEl = document.getElementById('response');
const goBtn = document.getElementById('go');
const classifyOpt = document.getElementById('classify');

function fmtBytes(n) {
  if (n < 1024) return n + ' B';
  if (n < 1024*1024) return (n/1024).toFixed(1) + ' KB';
  return (n/1024/1024).toFixed(2) + ' MB';
}
function fmtRp(n) {
  return 'Rp ' + n.toLocaleString('id-ID', { maximumFractionDigits: 2, minimumFractionDigits: 2 });
}

filesIn.addEventListener('change', () => {
  list.innerHTML = '';
  if (filesIn.files.length === 0) {
    drop.classList.remove('has-files');
    drop.querySelector('.primary').textContent = 'Click to choose PDFs';
    return;
  }
  drop.classList.add('has-files');
  drop.querySelector('.primary').textContent = `${filesIn.files.length} file(s) selected — click to change`;
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
  const classify = classifyOpt.checked;
  status.className = '';
  status.textContent = `Uploading ${filesIn.files.length} file(s)…`;
  summary.innerHTML = '';
  responseEl.textContent = '';
  goBtn.disabled = true;
  const t0 = performance.now();
  try {
    const r = await fetch(`/api/v1/mutations/extract-batch?classify=${classify}`, { method: 'POST', body: fd });
    const data = await r.json();
    const dt = ((performance.now() - t0) / 1000).toFixed(2);
    if (!r.ok) {
      status.className = 'err';
      status.textContent = `HTTP ${r.status} in ${dt}s — ${data.detail || 'error'}`;
      responseEl.textContent = JSON.stringify(data, null, 2);
      return;
    }
    status.className = 'ok';
    status.textContent = `HTTP ${r.status} in ${dt}s — processed ${data.audit.files_processed} file(s), ${data.audit.transactions_total} transactions, ${data.audit.credits_total} credits`;
    renderSummary(data);
    responseEl.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    status.className = 'err';
    status.textContent = `Network error: ${err}`;
  } finally {
    goBtn.disabled = false;
  }
});

function esc(s) {
  if (s == null) return '';
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

const CATEGORIES = ['Gaji', 'THR', 'Bonus', 'Insentif', 'Lainnya'];

function renderSummary(data) {
  const totals = data.audit.category_totals || {};
  const credits = data.credits || [];
  // Group every classified credit by its category (null → Lainnya, matching backend).
  const byCat = Object.fromEntries(CATEGORIES.map(c => [c, []]));
  for (const c of credits) {
    const cat = (c.category || 'Lainnya');
    (byCat[cat] || byCat.Lainnya).push(c);
  }
  // Sort each group by date ascending so the table reads chronologically.
  for (const k of Object.keys(byCat)) byCat[k].sort((a, b) => a.tanggal.localeCompare(b.tanggal));

  const sections = CATEGORIES.map(cat => {
    const t = totals[cat] || { count: 0, sum: 0, min: null };
    const items = byCat[cat];
    // Expand the high-signal categories by default; collapse Lainnya (usually noisy).
    const openByDefault = cat !== 'Lainnya' && items.length > 0;
    const minLabel = t.min == null ? '—' : fmtRp(t.min);
    const body = items.length === 0
      ? `<div class="empty">No transactions classified as ${cat}.</div>`
      : `<div class="tx-wrap"><table class="tx-table">
           <thead><tr>
             <th>Date</th>
             <th>Source file</th>
             <th class="num">Amount</th>
             <th>Description</th>
             <th class="num">Conf.</th>
             <th>Reason</th>
           </tr></thead>
           <tbody>${items.map(c => `
             <tr>
               <td>${esc(c.tanggal)}</td>
               <td class="src">${esc(c.source_file)}</td>
               <td class="num">${fmtRp(c.amount)}</td>
               <td class="desc">${esc(c.keterangan)}</td>
               <td class="num">${c.confidence == null ? '—' : c.confidence.toFixed(2)}</td>
               <td class="reason">${esc(c.reason || '')}</td>
             </tr>`).join('')}</tbody>
         </table></div>`;
    return `
      <details class="cat" data-category="${cat}" ${openByDefault ? 'open' : ''}>
        <summary>
          <span></span>
          <span class="cat-name">${cat}</span>
          <span class="cat-meta">${t.count} tx<span class="sep">·</span>min ${minLabel}</span>
          <span class="cat-sum">${fmtRp(t.sum)}</span>
        </summary>
        ${body}
      </details>`;
  }).join('');

  summary.innerHTML = `
    <div class="card" style="padding: 12px;">
      <h2 style="margin: 4px 8px 12px; font-size: 16px;">Totals by category — count · minimum · sum. Click a row for every transaction.</h2>
      ${sections}
    </div>`;
}
</script>
</body>
</html>"""


@app.get("/upload", response_class=HTMLResponse, include_in_schema=False)
def upload_page() -> HTMLResponse:
    """A simple multi-file upload page that bypasses Swagger UI's array
    renderer (which insists on one slot per file). Uses a native
    `<input type="file" multiple>` so the OS file picker handles
    multi-selection in a single click."""
    return HTMLResponse(_UPLOAD_PAGE)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post(
    "/api/v1/mutations/extract",
    response_model=ExtractionResponse,
    tags=["mutasi"],
    summary="Extract and classify transactions from a BCA mutation PDF",
)
async def extract_mutations(
    file: UploadFile = File(..., description="BCA Rekening Tahapan PDF"),
    classify: bool = Query(True, description="Run LLM classification on credit rows."),
    password: str | None = Form(
        None,
        description=(
            "Optional PDF password if the file is encrypted. "
            "Common patterns for Indonesian e-statements: last 6 digits of "
            "account number / birthdate DDMMYYYY / NIK. Leave blank for "
            "unencrypted PDFs."
        ),
    ),
) -> ExtractionResponse:
    settings = get_settings()

    if file.content_type not in {"application/pdf", "application/octet-stream"} \
            and not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload must be a PDF.")

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(data) > settings.max_pdf_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"PDF exceeds {settings.max_pdf_bytes} bytes",
        )

    try:
        result = run_pipeline(data, classify=classify, password=password)
    except PdfPasswordRequiredError as exc:
        # Distinct status detail so the client can prompt the user for a
        # password instead of treating it like a corrupt file.
        logger.warning("rejected upload (PdfPasswordRequiredError): %s", exc)
        raise HTTPException(
            status_code=422,
            detail=(
                "PDF is password-protected. Pass the password as the 'password' "
                "form field. Indonesian e-statements commonly use the account "
                "number's last 6 digits, the birthdate in DDMMYYYY, or NIK."
            ),
        ) from exc
    except (InvalidPdfError, UnsupportedBankError) as exc:
        # Client-side fault — the upload itself was the problem. Single-line
        # warning, no traceback; the client gets the message in the response.
        logger.warning("rejected upload (%s): %s", type(exc).__name__, exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        # Genuine server bug — keep the traceback for ops.
        logger.exception("unexpected pipeline failure")
        raise HTTPException(status_code=500, detail="Internal error while parsing PDF") from exc

    if result.audit.rows_detected == 0:
        raise HTTPException(status_code=422, detail="No transactions detected — is this a supported bank PDF?")

    return result


@app.post(
    "/api/v1/mutations/extract-batch",
    response_model=BatchExtractionResponse,
    tags=["mutasi"],
    summary="Upload MULTIPLE PDFs together and classify a year of credits in one LLM call",
    description=(
        "Accepts **multiple** BCA/BRI mutation PDFs in a single request. Each "
        "PDF is parsed independently, then ALL credit rows from ALL uploads "
        "are sent to one LLM call so the model can see recurring monthly "
        "payroll deposits — the key Gaji signal that's invisible when "
        "classifying one PDF at a time.\n\n"
        "### How to upload multiple files\n\n"
        "**In Swagger UI (this page):** click *Try it out*, then click *Choose Files* "
        "(plural). In the OS file picker, hold Ctrl (Windows/Linux) or Cmd (macOS) and "
        "select every PDF you want — they'll all upload in one request.\n\n"
        "**With curl:** pass `-F files=@path/to/file.pdf` once per file:\n"
        "```bash\n"
        "curl -X POST \\\n"
        "  -F files=@Mutasi_Januari_2026.pdf \\\n"
        "  -F files=@Mutasi_Februari_2026.pdf \\\n"
        "  -F files=@Mutasi_Maret_2026.pdf \\\n"
        "  http://localhost:8000/api/v1/mutations/extract-batch\n"
        "```\n\n"
        "Or expand a folder in one go:\n"
        "```bash\n"
        "curl -X POST $(for f in mutasi_haswin/*.pdf; do echo -n \"-F files=@$f \"; done) \\\n"
        "  http://localhost:8000/api/v1/mutations/extract-batch\n"
        "```\n\n"
        "**With Python (`httpx`):**\n"
        "```python\n"
        "import glob, httpx\n"
        "paths = sorted(glob.glob('mutasi_haswin/*.pdf'))\n"
        "files = [('files', (p.split('/')[-1], open(p,'rb'), 'application/pdf'))\n"
        "         for p in paths]\n"
        "r = httpx.post('http://localhost:8000/api/v1/mutations/extract-batch',\n"
        "               files=files, timeout=120)\n"
        "print(r.json()['audit']['category_totals'])\n"
        "```"
    ),
)
async def extract_mutations_batch(
    files: List[UploadFile] = File(
        ...,
        description=(
            "Upload ONE OR MORE PDFs. In Swagger UI, click 'Choose Files' and hold "
            "Ctrl (Windows/Linux) or Cmd (macOS) in the file picker to multi-select. "
            "With curl, repeat `-F files=@<path>` once per file. All credits across "
            "all files are classified together in a single LLM call."
        ),
    ),
    classify: bool = Query(True, description="Run cross-month LLM classification on credits."),
    password: str | None = Form(
        None,
        description=(
            "Optional PDF password applied to EVERY file in the batch. "
            "If different files have different passwords, upload them in "
            "separate requests."
        ),
    ),
) -> BatchExtractionResponse:
    settings = get_settings()
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    payload: list[tuple[str, bytes]] = []
    for f in files:
        if f.content_type not in {"application/pdf", "application/octet-stream"} \
                and not (f.filename or "").lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{f.filename!r}: not a PDF.")
        data = await f.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"{f.filename!r}: empty upload.")
        if len(data) > settings.max_pdf_bytes:
            raise HTTPException(status_code=413, detail=f"{f.filename!r}: exceeds {settings.max_pdf_bytes} bytes")
        payload.append((f.filename or "uploaded.pdf", data))

    try:
        result = run_pipeline_batch(payload, classify=classify, password=password)
    except PdfPasswordRequiredError as exc:
        logger.warning("rejected batch (PdfPasswordRequiredError): %s", exc)
        raise HTTPException(
            status_code=422,
            detail=(
                "One or more PDFs in the batch are password-protected. Pass the "
                "password as the 'password' form field. If files have different "
                "passwords, upload them in separate requests."
            ),
        ) from exc
    except (InvalidPdfError, UnsupportedBankError) as exc:
        logger.warning("rejected batch (%s): %s", type(exc).__name__, exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("unexpected batch pipeline failure")
        raise HTTPException(status_code=500, detail="Internal error while parsing batch") from exc

    if result.audit.transactions_total == 0:
        raise HTTPException(status_code=422, detail="No transactions detected in any file.")
    return result
