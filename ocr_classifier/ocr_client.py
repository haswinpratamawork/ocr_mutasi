"""Client for the PaddleOCR markdown service + text extraction.

We send a document to the Google-Cloud PaddleOCR endpoint and pull the
recognised text out of its `parsing_res_list[*].block_content`. The text
is what we hand to the LLM for classification — we never look at pixels
ourselves.

The text-extraction logic is split into a pure helper
(`extract_text_from_payload`) so it can be unit-tested without the network,
mirroring how the sibling services keep parsing logic side-effect free.
"""
from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from .config import get_settings
from .models import OcrAudit

logger = logging.getLogger(__name__)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_BLANKLINE_RE = re.compile(r"\n{3,}")


# ----------------------------- error types ---------------------------------

class OcrError(RuntimeError):
    """Base class for all OCR upstream failures."""


class OcrUnreachableError(OcrError):
    """The OCR service refused the connection (network/DNS error)."""


class OcrTimeoutError(OcrError):
    """The OCR service did not respond within the timeout."""


class OcrHttpError(OcrError):
    """The OCR service answered with an error (HTTP >=400 or an error body)."""

    def __init__(self, status_code: int | None, body: str) -> None:
        super().__init__(f"OCR service error {status_code}: {body[:300]}")
        self.status_code = status_code
        self.body = body


# ----------------------------- pure extraction ------------------------------

def _normalize_pages(json_result: Any) -> list[dict]:
    """Normalize `data.json_result` to a list of page objects.

    The sample response shows a single page object (a dict). A multi-page PDF
    is expected to return a list of such objects. Anything else → no pages.
    """
    if isinstance(json_result, dict):
        return [json_result]
    if isinstance(json_result, list):
        return [p for p in json_result if isinstance(p, dict)]
    return []


def _strip_html(markdown: str) -> str:
    """Drop HTML tags from the `data.markdown` fallback and collapse spaces."""
    text = _HTML_TAG_RE.sub(" ", markdown)
    return re.sub(r"[ \t]+", " ", text).strip()


def extract_text_from_payload(payload: dict, max_chars: int) -> tuple[str, int]:
    """Pull classification text out of an OCR response.

    Returns ``(text, page_count)``. Text is the concatenation of every
    ``block_content`` across all pages, joined by newlines, lightly
    whitespace-normalized and truncated to ``max_chars``. If no block content
    is present, falls back to the HTML-stripped ``data.markdown`` field.
    """
    data = payload.get("data") or {}
    pages = _normalize_pages(data.get("json_result"))

    blocks: list[str] = []
    for page in pages:
        for block in page.get("parsing_res_list") or []:
            if not isinstance(block, dict):
                continue
            content = (block.get("block_content") or "").strip()
            if content:
                blocks.append(content)

    text = "\n".join(blocks)
    if not text:
        # Fallback: the markdown field (often just an <img> wrapper, but
        # occasionally carries text when parsing_res_list is empty).
        text = _strip_html(data.get("markdown") or "")

    text = _MULTI_BLANKLINE_RE.sub("\n\n", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars]
    return text, len(pages)


# ----------------------------- network call ---------------------------------

async def extract_text(file_bytes: bytes, filename: str) -> tuple[str, OcrAudit]:
    """POST a document to the OCR service and return ``(text, OcrAudit)``.

    Raises:
        OcrUnreachableError: connection refused / DNS failure.
        OcrTimeoutError: the request timed out.
        OcrHttpError: the service answered >=400 or with an error body.
    """
    settings = get_settings()
    params = {"skip_orientation": str(settings.ocr_skip_orientation).lower()}
    headers = {"X-API-Key": settings.ocr_api_key}
    files = {"file": (filename or "document.pdf", file_bytes, "application/pdf")}

    try:
        async with httpx.AsyncClient(timeout=settings.ocr_timeout_s) as client:
            response = await client.post(
                settings.ocr_endpoint_url,
                params=params,
                headers=headers,
                files=files,
            )
    except httpx.TimeoutException as exc:
        raise OcrTimeoutError(f"OCR service timed out: {exc}") from exc
    except httpx.ConnectError as exc:
        raise OcrUnreachableError(
            f"OCR service not reachable at {settings.ocr_endpoint_url}: {exc}"
        ) from exc
    except httpx.HTTPError as exc:  # any other transport-level failure
        raise OcrUnreachableError(f"OCR request failed: {exc}") from exc

    if response.status_code >= 400:
        raise OcrHttpError(response.status_code, response.text)

    payload = response.json()
    # The service can answer HTTP 200 yet signal a failure in the body.
    response_code = payload.get("response_code")
    error_message = payload.get("error_message")
    if (response_code is not None and response_code != 200) or error_message:
        raise OcrHttpError(response_code, error_message or "OCR returned an error")

    text, page_count = extract_text_from_payload(payload, settings.max_classify_chars)
    audit = OcrAudit(
        ocr_request_id=payload.get("request_id"),
        ocr_response_time_ms=payload.get("response_time_ms"),
        extracted_char_count=len(text),
        page_count=page_count,
    )
    return text, audit
