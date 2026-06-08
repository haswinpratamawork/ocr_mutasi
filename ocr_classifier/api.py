"""FastAPI surface for the document-type classifier.

Exposes /classify (single) and /classify-batch (many) plus a health check
and a browser upload page. Keep this thin — all logic lives in `pipeline`.

Run from the repo root:  uvicorn ocr_classifier.api:app --port 8300 --reload
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse, RedirectResponse

from . import __version__
from .config import get_settings
from .models import BatchClassificationResponse, ClassificationResult
from .ocr_client import OcrHttpError, OcrTimeoutError, OcrUnreachableError
from .pipeline import run as run_pipeline, run_batch as run_pipeline_batch

logger = logging.getLogger("ocr_classifier.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="OCR Classifier",
    version=__version__,
    description=(
        "Classify a document (PDF) as one of ktp / kk / sk / slip / mutasi "
        "(or unknown). Runs the file through the PaddleOCR markdown service "
        "and asks an LLM to label the recognised text."
    ),
)
# Emit OpenAPI 3.0.3 so Swagger UI renders the multi-file upload as a real
# file picker (see _custom_openapi below) — same fix as ocr_mutasi.
app.openapi_version = "3.0.3"


def _custom_openapi() -> dict:
    """Rewrite 3.1-only binary-field syntax to 3.0-native `format: binary`
    so Swagger UI shows a file picker instead of a plain-text widget."""
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
            for value in node.values():
                rewrite(value)
        elif isinstance(node, list):
            for item in node:
                rewrite(item)

    rewrite(schema)
    app.openapi_schema = schema
    return schema


app.openapi = _custom_openapi


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect the bare URL to the upload page."""
    return RedirectResponse(url="/upload", status_code=307)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


_UPLOAD_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>OCR Classifier — Upload</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root { --fg:#1a1a1a; --muted:#6b7280; --bg:#f7f7f8; --panel:#fff; --border:#e5e7eb;
            --accent:#2563eb; --accent-hover:#1d4ed8; --ok:#059669; --err:#dc2626;
            --code-bg:#0f172a; --code-fg:#e2e8f0; }
    * { box-sizing:border-box; }
    body { margin:0; padding:24px; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
           background:var(--bg); color:var(--fg); }
    .container { max-width:980px; margin:0 auto; }
    h1 { margin:0 0 4px; font-size:22px; }
    p.tagline { margin:0 0 24px; color:var(--muted); font-size:14px; }
    .card { background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:20px; margin-bottom:16px; }
    label.file-drop { display:block; border:2px dashed var(--border); border-radius:8px; padding:32px 16px;
                      text-align:center; cursor:pointer; transition:border-color .15s, background .15s; }
    label.file-drop:hover { border-color:var(--accent); background:#f0f7ff; }
    label.file-drop.has-files { border-color:var(--ok); background:#f0fdf4; }
    label.file-drop .icon { font-size:32px; line-height:1; }
    label.file-drop .primary { display:block; margin-top:8px; font-weight:600; }
    label.file-drop .secondary { display:block; margin-top:4px; color:var(--muted); font-size:13px; }
    input[type=file] { display:none; }
    #file-list { margin-top:12px; padding:0; list-style:none; font-size:13px; }
    #file-list li { padding:4px 0; color:var(--muted); }
    .controls { display:flex; gap:16px; align-items:center; margin-top:16px; flex-wrap:wrap; }
    button { font:inherit; font-weight:600; padding:10px 18px; border-radius:6px; border:none;
             background:var(--accent); color:#fff; cursor:pointer; }
    button:hover { background:var(--accent-hover); }
    button:disabled { background:#94a3b8; cursor:not-allowed; }
    label.opt { display:flex; gap:6px; align-items:center; font-size:14px; color:var(--muted); cursor:pointer; }
    #status { margin:8px 0; font-size:13px; min-height:18px; }
    #status.ok { color:var(--ok); } #status.err { color:var(--err); }
    table.results { width:100%; border-collapse:collapse; font-size:13px; margin-top:8px; }
    table.results th, table.results td { padding:8px 12px; text-align:left; border-bottom:1px solid var(--border); vertical-align:top; }
    table.results th { background:#fafafb; font-weight:600; color:var(--muted); font-size:12px;
                       text-transform:uppercase; letter-spacing:.03em; }
    .badge { display:inline-block; padding:2px 10px; border-radius:999px; font-weight:600; font-size:12px;
             text-transform:uppercase; letter-spacing:.03em; background:#eef2ff; color:#3730a3; }
    .badge.unknown { background:#f1f5f9; color:#64748b; }
    .conf-high { color:var(--ok); font-weight:600; }
    .conf-medium { color:#f59e0b; font-weight:600; }
    .conf-low { color:var(--err); font-weight:600; }
    td.reason { color:var(--muted); max-width:360px; }
    td.err { color:var(--err); font-size:12px; }
    pre#response { background:var(--code-bg); color:var(--code-fg); padding:16px; border-radius:6px;
                   overflow:auto; font-size:12px; line-height:1.5; max-height:420px; margin:0; }
    details { margin-top:12px; } summary { cursor:pointer; color:var(--muted); font-size:13px; }
    a { color:var(--accent); }
  </style>
</head>
<body>
  <div class="container">
    <h1>OCR Classifier</h1>
    <p class="tagline">Select one or more documents. Each is sent to PaddleOCR and classified as
      <code>ktp</code> / <code>kk</code> / <code>sk</code> / <code>slip</code> / <code>mutasi</code> /
      <code>unknown</code>. Hold <kbd>Cmd</kbd>/<kbd>Ctrl</kbd> to multi-select. &nbsp;·&nbsp;
      <a href="/docs">Swagger UI</a></p>

    <form id="form" class="card">
      <label for="files" class="file-drop" id="drop">
        <span class="icon">📄</span>
        <span class="primary">Click to choose documents</span>
        <span class="secondary">Multi-select supported · PDF (images accepted too)</span>
        <input type="file" id="files" name="files" accept="application/pdf,.pdf,image/*" multiple>
      </label>
      <ul id="file-list"></ul>
      <div class="controls">
        <label class="opt"><input type="checkbox" id="include_text"> Include extracted OCR text in response</label>
        <button type="submit" id="go">Classify</button>
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
const includeText = document.getElementById('include_text');

function fmtBytes(n){ if(n<1024) return n+' B'; if(n<1048576) return (n/1024).toFixed(1)+' KB'; return (n/1048576).toFixed(2)+' MB'; }
function esc(s){ if(s==null) return ''; return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

filesIn.addEventListener('change', () => {
  list.innerHTML = '';
  if (filesIn.files.length === 0) {
    drop.classList.remove('has-files');
    drop.querySelector('.primary').textContent = 'Click to choose documents';
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
  if (filesIn.files.length === 0) { status.className='err'; status.textContent='Choose at least one file first.'; return; }
  const fd = new FormData();
  for (const f of filesIn.files) fd.append('files', f, f.name);
  status.className=''; status.textContent=`Uploading ${filesIn.files.length} file(s)… (OCR can take ~10s each)`;
  summary.innerHTML=''; responseEl.textContent=''; goBtn.disabled=true;
  const t0 = performance.now();
  try {
    const r = await fetch(`/classify-batch?include_text=${includeText.checked}`, { method:'POST', body:fd });
    const data = await r.json();
    const dt = ((performance.now()-t0)/1000).toFixed(2);
    if (!r.ok) {
      status.className='err'; status.textContent=`HTTP ${r.status} in ${dt}s — ${data.detail || 'error'}`;
      responseEl.textContent = JSON.stringify(data, null, 2); return;
    }
    status.className='ok'; status.textContent=`HTTP ${r.status} in ${dt}s — classified ${data.count} file(s)`;
    renderSummary(data);
    responseEl.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    status.className='err'; status.textContent=`Network error: ${err}`;
  } finally { goBtn.disabled=false; }
});

function renderSummary(data) {
  const rows = (data.results || []).map(r => {
    const conf = r.confidence || '';
    const errs = (r.audit && r.audit.errors || []).join('; ');
    return `<tr>
      <td>${esc(r.filename)}</td>
      <td><span class="badge ${r.document_type==='unknown'?'unknown':''}">${esc(r.document_type)}</span></td>
      <td class="conf-${esc(conf)}">${esc(conf)}</td>
      <td class="reason">${esc(r.reasoning)}</td>
      <td class="err">${esc(errs)}</td>
    </tr>`;
  }).join('');
  summary.innerHTML = `<div class="card" style="padding:12px;">
    <table class="results">
      <thead><tr><th>File</th><th>Type</th><th>Confidence</th><th>Reasoning</th><th>Notes</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>`;
}
</script>
</body>
</html>"""


@app.get("/upload", response_class=HTMLResponse, include_in_schema=False)
def upload_page() -> HTMLResponse:
    """A simple multi-file upload page (native multi-select picker)."""
    return HTMLResponse(_UPLOAD_PAGE)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "version": __version__}


def _read_and_validate(data: bytes, filename: str, max_bytes: int) -> None:
    if len(data) == 0:
        raise HTTPException(status_code=400, detail=f"Empty upload: {filename}")
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"{filename} exceeds {max_bytes} bytes")


@app.post(
    "/classify",
    response_model=ClassificationResult,
    tags=["classifier"],
    summary="Classify a single document",
)
async def classify_one(
    file: UploadFile = File(..., description="Document to classify (PDF; images accepted)."),
    include_text: bool = Query(False, description="Echo the extracted OCR text in the response."),
) -> ClassificationResult:
    settings = get_settings()
    data = await file.read()
    _read_and_validate(data, file.filename or "document.pdf", settings.max_pdf_bytes)
    try:
        return await run_pipeline(data, file.filename or "document.pdf", include_text=include_text)
    except OcrTimeoutError as exc:
        logger.warning("OCR timeout: %s", exc)
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except (OcrUnreachableError, OcrHttpError) as exc:
        logger.warning("OCR upstream error: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("unexpected classify failure")
        raise HTTPException(status_code=500, detail="Internal error while classifying document") from exc


@app.post(
    "/classify-batch",
    response_model=BatchClassificationResponse,
    tags=["classifier"],
    summary="Classify many documents in one request",
)
async def classify_many(
    files: list[UploadFile] = File(..., description="Documents to classify."),
    include_text: bool = Query(False, description="Echo the extracted OCR text in each result."),
) -> BatchClassificationResponse:
    settings = get_settings()
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > settings.max_files:
        raise HTTPException(status_code=413, detail=f"Too many files (max {settings.max_files}).")

    payload: list[tuple[str, bytes]] = []
    for upload in files:
        data = await upload.read()
        filename = upload.filename or "document.pdf"
        _read_and_validate(data, filename, settings.max_pdf_bytes)
        payload.append((filename, data))

    # Per-file OCR/LLM failures are captured inside each result's audit, so the
    # batch never raises for an individual bad document.
    results = await run_pipeline_batch(payload, include_text=include_text)
    return BatchClassificationResponse(count=len(results), results=results)
