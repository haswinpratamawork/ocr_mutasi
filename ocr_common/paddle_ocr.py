"""Shared client for the PaddleOCR markdown service.

This is the single OCR backend for the monorepo. `ocr_slip` and `ocr_sk` use it
as their parser-fallback OCR engine (replacing local Tesseract); `ocr_mutasi`
uses it for scanned bank statements. `ocr_classifier` has its own async client
but hits the same service.

The service is configured from the shared repo-root `.env`:
    OCR_ENDPOINT_URL   e.g. http://10.213.128.42:8060/predict/markdown
    OCR_API_KEY        sent as the `X-API-Key` header
    OCR_SKIP_ORIENTATION  "true"/"false" query flag (default false)
    OCR_TIMEOUT_S      request timeout in seconds (default 120)

It POSTs a PDF and returns the recognised text from
`data.json_result[*].parsing_res_list[*].block_content`. The parsing is split
into pure helpers so it can be unit-tested without the network.
"""
from __future__ import annotations

import io
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HTML_TAG_RE = re.compile(r"<[^>]+>")


# ----------------------------- error types ---------------------------------

class PaddleOcrError(RuntimeError):
    """Base class for OCR-service failures (subclasses RuntimeError so existing
    `except RuntimeError` fallback handlers keep working)."""


class PaddleOcrUnreachableError(PaddleOcrError):
    """The OCR service refused the connection (network/DNS error)."""


class PaddleOcrTimeoutError(PaddleOcrError):
    """The OCR service did not respond within the timeout."""


class PaddleOcrHttpError(PaddleOcrError):
    """The OCR service answered with an error (HTTP >=400 or an error body)."""


# ----------------------------- configuration --------------------------------

def _load_repo_env() -> None:
    """Best-effort load of the repo-root .env into os.environ (setdefault, so
    real environment variables win). Lets services that don't use
    pydantic-settings still pick up the OCR_* keys."""
    path = _REPO_ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _config() -> tuple[str, str, str, float]:
    _load_repo_env()
    url = os.environ.get("OCR_ENDPOINT_URL", "").strip()
    if not url:
        raise PaddleOcrError("OCR_ENDPOINT_URL is not set (configure it in the repo-root .env).")
    api_key = os.environ.get("OCR_API_KEY", "").strip()
    skip_orientation = os.environ.get("OCR_SKIP_ORIENTATION", "false").strip().lower()
    timeout = float(os.environ.get("OCR_TIMEOUT_S", "120") or "120")
    return url, api_key, skip_orientation, timeout


def _maybe_decrypt(pdf_bytes: bytes, password: str | None) -> bytes:
    """If a password is given, decrypt the PDF to a plain copy before upload
    (the OCR service can't open encrypted PDFs). Falls back to the original
    bytes if decryption isn't possible."""
    if not password:
        return pdf_bytes
    try:
        import pikepdf

        with pikepdf.open(io.BytesIO(pdf_bytes), password=password) as pdf:
            out = io.BytesIO()
            pdf.save(out)
            return out.getvalue()
    except Exception:
        return pdf_bytes


# ----------------------------- network call ---------------------------------

def fetch_payload(pdf_bytes: bytes, filename: str = "document.pdf", password: str | None = None) -> dict:
    """POST a PDF to the OCR service and return the parsed JSON payload.

    Raises:
        PaddleOcrUnreachableError, PaddleOcrTimeoutError, PaddleOcrHttpError.
    """
    url, api_key, skip_orientation, timeout = _config()
    data = _maybe_decrypt(pdf_bytes, password)
    try:
        response = requests.post(
            url,
            headers={"X-API-Key": api_key},
            params={"skip_orientation": skip_orientation},
            files={"file": (filename or "document.pdf", data, "application/pdf")},
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise PaddleOcrTimeoutError(f"OCR service timed out: {exc}") from exc
    except requests.ConnectionError as exc:
        raise PaddleOcrUnreachableError(f"OCR service not reachable at {url}: {exc}") from exc
    except requests.RequestException as exc:
        raise PaddleOcrUnreachableError(f"OCR request failed: {exc}") from exc

    if response.status_code >= 400:
        raise PaddleOcrHttpError(f"OCR service error {response.status_code}: {response.text[:300]}")

    payload = response.json()
    response_code = payload.get("response_code")
    error_message = payload.get("error_message")
    if (response_code is not None and response_code != 200) or error_message:
        raise PaddleOcrHttpError(f"OCR service error {response_code}: {error_message or 'unknown error'}")
    return payload


# ----------------------------- pure parsing ---------------------------------

def _normalize_pages(json_result: Any) -> list[dict]:
    if isinstance(json_result, dict):
        return [json_result]
    if isinstance(json_result, list):
        return [p for p in json_result if isinstance(p, dict)]
    return []


def _page_text(page: dict) -> str:
    blocks: list[str] = []
    for block in page.get("parsing_res_list") or []:
        if isinstance(block, dict):
            content = (block.get("block_content") or "").strip()
            if content:
                blocks.append(content)
    return "\n".join(blocks)


def _strip_html(markdown: str) -> str:
    return re.sub(r"[ \t]+", " ", _HTML_TAG_RE.sub(" ", markdown)).strip()


def _page_text_from_payload(payload: dict) -> str:
    """All recognised text from one OCR response (joins blocks across whatever
    pages it returned; falls back to the HTML-stripped markdown rendering)."""
    data = payload.get("data") or {}
    parts = [_page_text(p) for p in _normalize_pages(data.get("json_result"))]
    text = "\n".join(t for t in parts if t).strip()
    if not text:
        text = _strip_html(data.get("markdown") or "")
    return text


def _split_pdf_pages(pdf_bytes: bytes) -> list[bytes]:
    """Split a PDF into one single-page PDF (bytes) per page.

    The OCR service processes a single page per request, so we OCR each page
    separately — otherwise a multi-page document (e.g. a 3-month salary slip
    with one month per page) only yields page 1. Returns ``[]`` when the input
    can't be split (e.g. it's an image, or pypdfium2 is unavailable), so the
    caller can fall back to sending the whole file.
    """
    try:
        import pypdfium2 as pdfium
    except Exception:
        return []
    src = None
    try:
        src = pdfium.PdfDocument(pdf_bytes)
        blobs: list[bytes] = []
        for index in range(len(src)):
            dst = pdfium.PdfDocument.new()
            dst.import_pages(src, [index])
            buf = io.BytesIO()
            dst.save(buf)
            dst.close()
            blobs.append(buf.getvalue())
        return blobs
    except Exception:
        return []
    finally:
        if src is not None:
            try:
                src.close()
            except Exception:
                pass


def pages_from_payload(payload: dict, filename: str = "document.pdf") -> dict:
    """Map an OCR response into the same `pages[]` structure the salary-slip /
    employment-letter parsers expect from their (old) Tesseract extractor."""
    data = payload.get("data") or {}
    raw_pages = _normalize_pages(data.get("json_result"))

    pages: list[dict] = []
    for index, raw in enumerate(raw_pages):
        text = _page_text(raw)
        pages.append(
            {
                "page_number": index + 1,
                "char_count": len(text),
                "text": text,
                "lines": [ln.strip() for ln in text.splitlines() if ln.strip()],
            }
        )

    if not any(p["text"] for p in pages):
        # Fall back to the markdown rendering when block content is empty.
        markdown = _strip_html(data.get("markdown") or "")
        if markdown:
            pages = [
                {
                    "page_number": 1,
                    "char_count": len(markdown),
                    "text": markdown,
                    "lines": [ln.strip() for ln in markdown.splitlines() if ln.strip()],
                }
            ]

    return {
        "source_file": filename,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "extraction_method": "ocr_paddle",
        "page_count": len(pages),
        "pages": pages,
        "warnings": [f"OCR via PaddleOCR service (request_id={payload.get('request_id')})."],
    }


# ----------------------------- public API -----------------------------------

def extract_pages_from_bytes(pdf_bytes: bytes, filename: str = "document.pdf", password: str | None = None) -> dict:
    """OCR a PDF (bytes) and return one ``pages[]`` entry PER SOURCE PAGE.

    Because the OCR service handles a single page per request, we split the PDF
    and OCR each page separately, then assemble the per-page results. This is
    what makes a multi-page document (e.g. a 3-month salary slip, one month per
    page) extract every page instead of only the first.
    """
    decrypted = _maybe_decrypt(pdf_bytes, password)
    page_blobs = _split_pdf_pages(decrypted)

    request_ids: list[str] = []
    page_warnings: list[str] = []
    if page_blobs:
        # Each split page is a single-page PDF — name it with a real .pdf
        # extension (the OCR service validates the format from the filename).
        stem = Path(filename).stem or "document"
        page_texts: list[str] = []
        failures = 0
        for index, blob in enumerate(page_blobs):
            try:
                payload = fetch_payload(blob, filename=f"{stem}-page-{index + 1}.pdf")
            except PaddleOcrError as exc:
                # One page failing (timeout, transient 5xx, a bad page) must not
                # sink the whole document — keep the slot empty and carry on.
                failures += 1
                page_warnings.append(f"page {index + 1} OCR failed: {exc}")
                page_texts.append("")
                continue
            if payload.get("request_id"):
                request_ids.append(str(payload["request_id"]))
            page_texts.append(_page_text_from_payload(payload))
        if failures and failures == len(page_blobs):
            # Every page failed — that's a real OCR outage, surface it.
            raise PaddleOcrError("; ".join(page_warnings) or "OCR failed for every page")
    else:
        # Couldn't split (an image, or not a PDF) — OCR the whole file once.
        payload = fetch_payload(decrypted, filename=filename)
        if payload.get("request_id"):
            request_ids.append(str(payload["request_id"]))
        page_texts = [_page_text_from_payload(payload)]

    pages = [
        {
            "page_number": i + 1,
            "char_count": len(text),
            "text": text,
            "lines": [ln.strip() for ln in text.splitlines() if ln.strip()],
        }
        for i, text in enumerate(page_texts)
    ]
    return {
        "source_file": filename,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "extraction_method": "ocr_paddle",
        "page_count": len(pages),
        "pages": pages,
        "warnings": [f"OCR via PaddleOCR service ({len(pages)} page(s); request_ids={request_ids})."]
        + page_warnings,
    }


def extract_pages(pdf_path: Path | str, password: str | None = None) -> dict:
    """OCR a PDF on disk and return the `pages[]` extraction dict.

    Drop-in replacement for the old `TesseractOcrExtractor().extract(...)`.
    """
    path = Path(pdf_path)
    return extract_pages_from_bytes(path.read_bytes(), filename=path.name, password=password)


def extract_text_from_bytes(pdf_bytes: bytes, password: str | None = None) -> str:
    """OCR a PDF (bytes) and return all recognised text as one string (every
    page, joined by blank lines). Used by ocr_mutasi's scanned-statement path.
    """
    pages = extract_pages_from_bytes(pdf_bytes, password=password)["pages"]
    return "\n\n".join(p["text"] for p in pages if p["text"]).strip()
