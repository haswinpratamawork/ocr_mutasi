"""FastAPI surface for ocr_match.

Two business endpoints (``GET /health``, ``POST /api/v1/match``) plus a
handful of usability routes (root redirect, ``/upload`` HTML page,
``/favicon.ico``). The HTTP layer is intentionally thin — all behaviour
lives in ``pipeline.run``.

The OpenAPI schema is downgraded to 3.0.3 and post-processed so file
fields render as native file pickers in Swagger UI, matching the same
trick used by ocr_mutasi and ocr_slip.
"""
from __future__ import annotations

import logging
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse, RedirectResponse

from . import __version__
from .config import get_settings
from .models import MatchResponse
from .pipeline import run as run_pipeline
from .upstream import UpstreamHttpError, UpstreamUnreachableError

logger = logging.getLogger("ocr_match.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="OCR Match",
    version=__version__,
    description=(
        "Pair Indonesian salary slips with bank-credit Gaji rows. "
        "Calls ocr_slip and ocr_mutasi upstream; runs an LLM matcher per month."
    ),
)
app.openapi_version = "3.0.3"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _custom_openapi() -> dict:
    """Same OpenAPI 3.1 → 3.0.3 binary-field patch as the sibling services."""
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


# --------------------------- non-API routes -------------------------------

@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/upload", status_code=307)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


# --------------------------- API routes -----------------------------------

@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post(
    "/api/v1/match",
    response_model=MatchResponse,
    tags=["match"],
    summary="Pair salary slips with bank-credit Gaji rows",
    description=(
        "Accepts two file groups: `slips` (salary slip PDFs) and `mutations` "
        "(bank statement PDFs). Slips are parsed by ocr_slip:/parse, mutations "
        "are parsed AND classified by ocr_mutasi:/api/v1/mutations/extract-batch. "
        "Only credits with `category == \"Gaji\"` are considered for matching. "
        "An LLM call per month assigns each slip to at most one credit, with a "
        "≤15%% amount tolerance and fuzzy clinic-name matching."
    ),
)
async def match_endpoint(
    slips: List[UploadFile] = File(..., description="Salary-slip PDFs."),
    mutations: List[UploadFile] = File(..., description="Bank-statement PDFs."),
    slip_password: str | None = Form(
        None,
        description=(
            "Optional PDF password applied to every salary-slip PDF in the "
            "request. Slips and bank statements typically use different "
            "passwords; specify them independently."
        ),
    ),
    mutation_password: str | None = Form(
        None,
        description="Optional PDF password applied to every bank-statement PDF.",
    ),
) -> MatchResponse:
    settings = get_settings()
    if not slips:
        raise HTTPException(status_code=400, detail="Upload at least one salary slip PDF.")
    if not mutations:
        raise HTTPException(status_code=400, detail="Upload at least one bank-statement PDF.")
    total = len(slips) + len(mutations)
    if total > settings.max_files:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files ({total} > MAX_FILES={settings.max_files}).",
        )

    async def _gather(group: List[UploadFile]) -> list[tuple[str, bytes]]:
        out: list[tuple[str, bytes]] = []
        for f in group:
            name = f.filename or "unnamed.pdf"
            if not (f.content_type in {"application/pdf", "application/octet-stream"}
                    or name.lower().endswith(".pdf")):
                raise HTTPException(status_code=400, detail=f"{name!r}: not a PDF.")
            data = await f.read()
            if not data:
                raise HTTPException(status_code=400, detail=f"{name!r}: empty upload.")
            out.append((name, data))
        return out

    slip_pdfs = await _gather(slips)
    mutation_pdfs = await _gather(mutations)

    try:
        return await run_pipeline(
            slip_pdfs,
            mutation_pdfs,
            slip_password=slip_password,
            mutation_password=mutation_password,
        )
    except UpstreamUnreachableError as exc:
        logger.warning("upstream unreachable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except UpstreamHttpError as exc:
        logger.warning("upstream error: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("unexpected pipeline failure")
        raise HTTPException(status_code=500, detail="Internal error while matching") from exc


# --------------------------- /upload HTML page ----------------------------

_UPLOAD_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>OCR Match — Upload</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root { --fg: #1a1a1a; --muted: #6b7280; --bg: #f7f7f8; --panel: #fff;
            --border: #e5e7eb; --accent: #2563eb; --accent-hover: #1d4ed8;
            --ok: #059669; --err: #dc2626; --warn: #f59e0b;
            --code-bg: #0f172a; --code-fg: #e2e8f0; }
    * { box-sizing: border-box; }
    body { margin: 0; padding: 24px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           background: var(--bg); color: var(--fg); }
    .container { max-width: 1100px; margin: 0 auto; }
    h1 { margin: 0 0 4px; font-size: 22px; }
    p.tagline { margin: 0 0 24px; color: var(--muted); font-size: 14px; }
    .card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
            padding: 20px; margin-bottom: 16px; }
    .drops { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    @media (max-width: 720px) { .drops { grid-template-columns: 1fr; } }
    label.file-drop { display: block; border: 2px dashed var(--border); border-radius: 8px;
                      padding: 28px 16px; text-align: center; cursor: pointer;
                      transition: border-color .15s, background .15s; }
    label.file-drop:hover { border-color: var(--accent); background: #f0f7ff; }
    label.file-drop.has-files { border-color: var(--ok); background: #f0fdf4; }
    label.file-drop .group-name { display:block; font-size:11px; letter-spacing:.05em;
                                   text-transform:uppercase; color: var(--muted); margin-bottom: 4px; }
    label.file-drop .icon { font-size: 28px; line-height: 1; }
    label.file-drop .primary { display: block; margin-top: 8px; font-weight: 600; }
    label.file-drop .secondary { display: block; margin-top: 4px; color: var(--muted); font-size: 12px; }
    input[type=file] { display: none; }
    .file-list { margin-top: 8px; padding: 0; list-style: none; font-size: 12px; color: var(--muted); }
    .file-list li { padding: 2px 0; }
    .pw-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 14px; }
    @media (max-width: 720px) { .pw-grid { grid-template-columns: 1fr; } }
    .pw-grid label { display: block; font-size: 12px; text-transform: uppercase;
                     letter-spacing: .04em; color: var(--muted); margin-bottom: 4px; }
    .pw-grid input[type=password] { width: 100%; padding: 8px 10px; border-radius: 6px;
                                    border: 1px solid var(--border); font: inherit; font-size: 13px; }
    .pw-grid .hint { font-size: 12px; color: var(--muted); margin-top: 4px; }
    .controls { display: flex; gap: 16px; align-items: center; margin-top: 16px; flex-wrap: wrap; }
    button { font: inherit; font-weight: 600; padding: 10px 18px; border-radius: 6px;
             border: none; background: var(--accent); color: #fff; cursor: pointer; }
    button:hover { background: var(--accent-hover); }
    button:disabled { background: #94a3b8; cursor: not-allowed; }
    #status { margin: 8px 0; font-size: 13px; min-height: 18px; }
    #status.ok { color: var(--ok); } #status.err { color: var(--err); }
    /* Match cards */
    #matches h2, #unmatched h2, #upstream-errors h2 { margin: 0 0 12px; font-size: 16px; }
    .pair { display: grid; grid-template-columns: 1fr 24px 1fr; gap: 12px; align-items: center;
            padding: 14px 16px; border: 1px solid var(--border); border-radius: 8px;
            margin-bottom: 10px; background: var(--panel); }
    .pair .arrow { text-align: center; color: var(--ok); font-size: 22px; font-weight: 700; }
    .pair .side .label { font-size: 11px; letter-spacing:.05em; text-transform:uppercase;
                         color: var(--muted); margin-bottom: 4px; display: block; }
    .pair .side .title { font-weight: 600; font-size: 14px; }
    .pair .side .meta { font-size: 12px; color: var(--muted); margin-top: 2px;
                        word-break: break-all; }
    .pair .side .amount { font-variant-numeric: tabular-nums; font-weight: 600;
                          font-size: 15px; margin-top: 6px; }
    .pair .footer { grid-column: 1 / -1; margin-top: 8px; padding-top: 8px;
                    border-top: 1px solid var(--border); font-size: 12px; color: var(--muted); }
    .pair .footer .conf { font-weight: 600; color: var(--fg); }
    .pair .footer .diff.pos { color: var(--ok); } .pair .footer .diff.neg { color: var(--err); }
    .pair .footer .reason { margin-top: 4px; font-style: italic; }
    .pair .footer .badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
                           font-size: 11px; font-weight: 600; letter-spacing: .04em;
                           text-transform: uppercase; }
    .pair .footer .badge.pattern-next_month { background: #dbeafe; color: #1e40af; }
    .pair .footer .badge.pattern-same_month { background: #f3e8ff; color: #6b21a8; }
    /* Unmatched lists */
    #unmatched .group { background: var(--panel); border: 1px solid var(--border);
                        border-radius: 8px; padding: 12px 16px; margin-bottom: 10px; }
    #unmatched .group h3 { margin: 0 0 8px; font-size: 14px; color: var(--warn); }
    #unmatched ul { margin: 0; padding-left: 16px; font-size: 13px; }
    #unmatched li { padding: 2px 0; color: var(--muted); }
    /* Audit / errors */
    #audit { background: #fafafb; border: 1px solid var(--border); border-radius: 8px;
             padding: 12px 16px; margin-bottom: 16px; }
    #audit .row { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 8px;
                  font-size: 13px; }
    #audit .row .cell { padding: 8px 10px; background: var(--panel); border-radius: 6px;
                       border: 1px solid var(--border); }
    #audit .cell .label { display:block; font-size: 11px; letter-spacing:.04em;
                          text-transform:uppercase; color: var(--muted); margin-bottom: 2px; }
    #audit .cell .value { font-weight: 600; font-variant-numeric: tabular-nums; }
    #upstream-errors { background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px;
                       padding: 14px 18px; margin-top: 12px; }
    #upstream-errors li { color: var(--err); }
    /* Raw-JSON toggle */
    details { margin-top: 12px; }
    summary { cursor: pointer; color: var(--muted); font-size: 13px; }
    pre#response { background: var(--code-bg); color: var(--code-fg); padding: 16px; border-radius: 6px;
                   overflow: auto; font-size: 12px; line-height: 1.5; max-height: 480px; margin: 0; }
    a { color: var(--accent); }
  </style>
</head>
<body>
  <div class="container">
    <h1>OCR Match — Salary slip ↔ bank mutation</h1>
    <p class="tagline">Upload your salary slips on the left and bank statements on the right. The matcher pairs them by month, amount (±15%) and fuzzy company-name match (e.g. <em>Alsut</em> → <em>Alam Sutera</em>). &nbsp;·&nbsp; <a href="/docs">Swagger UI</a></p>

    <form id="form" class="card">
      <div class="drops">
        <label for="slips" class="file-drop" id="drop-s">
          <span class="group-name">Salary slips</span>
          <span class="icon">🧾</span>
          <span class="primary">Click to choose slip PDFs</span>
          <span class="secondary">Multi-select supported</span>
          <input type="file" id="slips" name="slips" accept="application/pdf,.pdf" multiple>
          <ul class="file-list" id="list-s"></ul>
        </label>
        <label for="mutations" class="file-drop" id="drop-m">
          <span class="group-name">Bank statements</span>
          <span class="icon">🏦</span>
          <span class="primary">Click to choose bank-statement PDFs</span>
          <span class="secondary">Multi-select supported</span>
          <input type="file" id="mutations" name="mutations" accept="application/pdf,.pdf" multiple>
          <ul class="file-list" id="list-m"></ul>
        </label>
      </div>
      <div class="pw-grid">
        <div>
          <label for="slip_password">Slip-PDF password (optional)</label>
          <input type="password" id="slip_password" name="slip_password" placeholder="Leave blank for unencrypted slips">
          <div class="hint">Usually employee ID, NIK, or birthdate.</div>
        </div>
        <div>
          <label for="mutation_password">Bank-statement password (optional)</label>
          <input type="password" id="mutation_password" name="mutation_password" placeholder="Leave blank for unencrypted statements">
          <div class="hint">Usually account-no last 6 digits, DDMMYYYY, or NIK.</div>
        </div>
      </div>

      <div class="controls">
        <button type="submit" id="go">Match</button>
      </div>
      <div id="status"></div>
    </form>

    <div id="audit-wrap"></div>
    <div id="matches"></div>
    <div id="unmatched"></div>
    <div id="upstream-errors" style="display:none"></div>

    <details>
      <summary>Show raw JSON response</summary>
      <pre id="response">(no request sent yet)</pre>
    </details>
  </div>

<script>
const form = document.getElementById('form');
const slipsIn = document.getElementById('slips');
const mutsIn = document.getElementById('mutations');
const dropS = document.getElementById('drop-s');
const dropM = document.getElementById('drop-m');
const listS = document.getElementById('list-s');
const listM = document.getElementById('list-m');
const status = document.getElementById('status');
const auditWrap = document.getElementById('audit-wrap');
const matchesEl = document.getElementById('matches');
const unmatchedEl = document.getElementById('unmatched');
const upstreamErrors = document.getElementById('upstream-errors');
const responseEl = document.getElementById('response');
const goBtn = document.getElementById('go');

function esc(s) {
  if (s == null) return '';
  return String(s).replace(/[&<>"']/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function fmtRp(n) {
  if (n == null) return '—';
  const sign = n < 0 ? '-' : '';
  return sign + 'Rp ' + Math.abs(Number(n)).toLocaleString('id-ID',
    { maximumFractionDigits: 2, minimumFractionDigits: 0 });
}
function bindDrop(input, drop, list, label) {
  input.addEventListener('change', () => {
    list.innerHTML = '';
    if (input.files.length === 0) {
      drop.classList.remove('has-files');
      drop.querySelector('.primary').textContent = label;
      return;
    }
    drop.classList.add('has-files');
    drop.querySelector('.primary').textContent =
      `${input.files.length} file(s) selected — click to change`;
    for (const f of input.files) {
      const li = document.createElement('li');
      li.textContent = `• ${f.name}`;
      list.appendChild(li);
    }
  });
}
bindDrop(slipsIn, dropS, listS, 'Click to choose slip PDFs');
bindDrop(mutsIn, dropM, listM, 'Click to choose bank-statement PDFs');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (slipsIn.files.length === 0 || mutsIn.files.length === 0) {
    status.className = 'err';
    status.textContent = 'Need at least one file in each group.';
    return;
  }
  const fd = new FormData();
  for (const f of slipsIn.files) fd.append('slips', f, f.name);
  for (const f of mutsIn.files)  fd.append('mutations', f, f.name);
  const sp = document.getElementById('slip_password').value;
  const mp = document.getElementById('mutation_password').value;
  if (sp) fd.append('slip_password', sp);
  if (mp) fd.append('mutation_password', mp);

  status.className = '';
  status.textContent = `Matching ${slipsIn.files.length} slip(s) against ${mutsIn.files.length} statement(s)…`;
  auditWrap.innerHTML = '';
  matchesEl.innerHTML = '';
  unmatchedEl.innerHTML = '';
  upstreamErrors.style.display = 'none';
  upstreamErrors.innerHTML = '';
  responseEl.textContent = '';
  goBtn.disabled = true;
  const t0 = performance.now();
  try {
    const r = await fetch('/api/v1/match', { method: 'POST', body: fd });
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
      `HTTP ${r.status} in ${dt}s — ${data.audit.matched_count} of ${data.audit.slip_count} slips matched`
      + (data.audit.matcher_errors?.length ? `, ${data.audit.matcher_errors.length} matcher error(s)` : '');
    renderAudit(data);
    renderMatches(data);
    renderUnmatched(data);
    renderUpstreamErrors(data);
  } catch (err) {
    status.className = 'err';
    status.textContent = `Network error: ${err}`;
  } finally {
    goBtn.disabled = false;
  }
});

function renderAudit(data) {
  const a = data.audit;
  auditWrap.innerHTML = `
    <div id="audit">
      <div class="row">
        <div class="cell"><span class="label">Slips</span><span class="value">${a.slip_count}</span></div>
        <div class="cell"><span class="label">Gaji credits</span><span class="value">${a.credit_count}</span></div>
        <div class="cell"><span class="label">Matched</span><span class="value">${a.matched_count}</span></div>
        <div class="cell"><span class="label">Months processed</span><span class="value">${esc((a.months_processed||[]).join(', ')) || '—'}</span></div>
      </div>
    </div>`;
}

function renderMatches(data) {
  const items = data.matches || [];
  if (items.length === 0) {
    matchesEl.innerHTML = `<div class="card" style="text-align:center;color:var(--muted)">No matches were produced.</div>`;
    return;
  }
  matchesEl.innerHTML = `<h2>${items.length} match${items.length === 1 ? '' : 'es'}</h2>` +
    items.map(p => {
      const sign = p.amount_diff_rp >= 0 ? '+' : '';
      const cls  = p.amount_diff_rp >= 0 ? 'pos' : 'neg';
      const pct  = (p.amount_diff_pct * 100).toFixed(2);
      return `
      <div class="pair">
        <div class="side">
          <span class="label">Slip</span>
          <div class="title">${esc(p.slip.worker_name) || '(unknown)'}</div>
          <div class="meta">${esc(p.slip.institution_name) || '—'}<br>${esc(p.slip.source_file)}</div>
          <div class="amount">${fmtRp(p.slip.total_paid)}</div>
        </div>
        <div class="arrow">↔</div>
        <div class="side">
          <span class="label">Bank credit</span>
          <div class="title">${esc(p.credit.tanggal)}</div>
          <div class="meta">${esc(p.credit.keterangan)}<br>${esc(p.credit.source_file)}</div>
          <div class="amount">${fmtRp(p.credit.amount)}</div>
        </div>
        <div class="footer">
          <span class="badge pattern-${esc(p.match_pattern || 'same_month')}">${
            (p.match_pattern || 'same_month') === 'next_month' ? 'X+1' : 'same month'
          }</span>
          &nbsp;·&nbsp;
          <span class="conf">conf ${p.confidence.toFixed(2)}</span>
          &nbsp;·&nbsp;
          diff: <span class="diff ${cls}">${sign}${fmtRp(p.amount_diff_rp)}${p.slip.total_paid ? ` (${sign}${pct}%)` : ''}</span>
          <div class="reason">${esc(p.reason)}</div>
        </div>
      </div>`;
    }).join('');
}

function renderUnmatched(data) {
  const s = data.unmatched_slips || [];
  const c = data.unmatched_credits || [];
  if (s.length === 0 && c.length === 0) return;
  unmatchedEl.innerHTML = `<h2>Unmatched</h2>` +
    (s.length ? `
      <div class="group">
        <h3>${s.length} slip(s) without a bank credit</h3>
        <ul>${s.map(x => `<li><strong>${esc(x.source_file)}</strong> — ${fmtRp(x.total_paid)} ${esc(x.month) ? '('+esc(x.month)+')' : ''}</li>`).join('')}</ul>
      </div>` : '') +
    (c.length ? `
      <div class="group">
        <h3>${c.length} Gaji credit(s) without a slip</h3>
        <ul>${c.map(x => `<li><strong>${esc(x.tanggal)}</strong> — ${fmtRp(x.amount)} — ${esc(x.keterangan.slice(0,90))}</li>`).join('')}</ul>
      </div>` : '');
}

function renderUpstreamErrors(data) {
  const errs = (data.audit.upstream_errors || []).concat(data.audit.matcher_errors || []);
  if (errs.length === 0) return;
  upstreamErrors.style.display = 'block';
  upstreamErrors.innerHTML = `<h2>${errs.length} upstream/matcher error(s)</h2><ul>` +
    errs.map(e => `<li>${esc(e)}</li>`).join('') + `</ul>`;
}
</script>
</body>
</html>"""


@app.get("/upload", response_class=HTMLResponse, include_in_schema=False)
def upload_page() -> HTMLResponse:
    """Self-contained HTML page with two drop zones (slips, mutations)
    and a Match button. Bypasses Swagger UI's awkward array-of-array
    file editor."""
    return HTMLResponse(_UPLOAD_PAGE)
