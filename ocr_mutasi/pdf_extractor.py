"""Thin wrapper around pypdfium2 that yields positioned text chunks.

PDFium uses a bottom-left origin (Y grows upward). We convert to a top-left
origin (Y grows downward) here so every consumer downstream can reason about
"top of page" vs "bottom of page" naturally.

This module is the only place that imports pypdfium2 and pikepdf. Failures
from the underlying libraries are re-raised as ``InvalidPdfError`` or
``PdfPasswordRequiredError`` so callers never need to import third-party
exception types.

Password handling
-----------------
Indonesian bank e-statements are commonly user-password-protected (the
password is usually a known pattern like account-no last 6 digits, DOB,
or NIK). To open them we try in this order:

  1. The caller-supplied password (if any).
  2. The empty string ``""`` — handles owner-password-only PDFs and a
     surprising number of "encrypted" files with no real password.
  3. ``pikepdf`` fallback: open with whatever credentials we have, save an
     unencrypted in-memory copy, then re-open with pypdfium2. pikepdf
     supports more encryption variants than pypdfium2 alone and
     legitimately strips owner-only locks (the PDF spec considers them
     informational, not access control).

If all three fail with a password error, we raise
``PdfPasswordRequiredError`` so the API layer can return a clear ``422``
asking the caller to supply the password. We do NOT brute-force or
dictionary-attack — that's outside this service's scope.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, Optional, Union

import pikepdf
import pypdfium2 as pdfium

from .models import TextChunk

PdfInput = Union[str, Path, bytes]


class InvalidPdfError(ValueError):
    """The input bytes don't form a readable PDF (corrupt, truncated, not a PDF)."""


class PdfPasswordRequiredError(ValueError):
    """The PDF is password-protected and the caller didn't supply (or supplied
    the wrong) password. Distinct from :class:`InvalidPdfError` so the API
    layer can surface a 'please provide password' message instead of a
    generic 'could not read PDF'.
    """


def _looks_like_password_error(exc: BaseException) -> bool:
    """PDFium error messages aren't typed, so we match on text. Stable enough —
    PDFium's wording has been the same for years."""
    msg = str(exc).lower()
    return ("password" in msg) or ("incorrect password" in msg)


def _as_bytes(pdf_input: PdfInput) -> bytes:
    if isinstance(pdf_input, bytes):
        return pdf_input
    return Path(pdf_input).read_bytes()


def _open_with_pypdfium(pdf_input: PdfInput, password: Optional[str]) -> "pdfium.PdfDocument":
    if password is None:
        return pdfium.PdfDocument(pdf_input)
    return pdfium.PdfDocument(pdf_input, password=password)


def _open_with_pikepdf_fallback(pdf_bytes: bytes, password: Optional[str]) -> "pdfium.PdfDocument":
    """Open the PDF via pikepdf, write an unencrypted copy to memory, re-open
    that copy with pypdfium2 and return it.

    Raises:
        PdfPasswordRequiredError: if pikepdf also refuses with a password error.
        InvalidPdfError: for any other pikepdf failure.
    """
    try:
        with pikepdf.open(io.BytesIO(pdf_bytes), password=password or "") as pdf:
            out = io.BytesIO()
            pdf.save(out)  # default save drops encryption
        out.seek(0)
        return pdfium.PdfDocument(out.read())
    except pikepdf.PasswordError as exc:
        raise PdfPasswordRequiredError(
            "PDF is password-protected and the supplied password (if any) is incorrect"
        ) from exc
    except (pikepdf.PdfError, Exception) as exc:  # noqa: BLE001
        raise InvalidPdfError(f"Could not read PDF via pikepdf fallback: {exc}") from exc


def extract_chunks(pdf_input: PdfInput, password: Optional[str] = None) -> list[TextChunk]:
    """Open a PDF and return every text rect on every page as a TextChunk.

    Args:
        pdf_input: bytes, path, or path-like.
        password: optional user-supplied password to try first. ``None`` is
            fine for unencrypted PDFs.

    Each rect is one contiguous run of text as identified by PDFium's text
    page (roughly: a span of characters with the same style sitting on the
    same baseline). This granularity matches what we need for column-based
    table reconstruction.

    Raises:
        InvalidPdfError: when the input isn't a readable PDF (corrupt, truncated,
            wrong format).
        PdfPasswordRequiredError: when the PDF needs a password the caller
            didn't supply (or supplied wrong).
    """
    # Pass 1 — pypdfium2 with the caller's password (or none).
    doc: Optional[pdfium.PdfDocument] = None
    try:
        doc = _open_with_pypdfium(pdf_input, password)
    except pdfium.PdfiumError as first_exc:
        if not _looks_like_password_error(first_exc):
            raise InvalidPdfError(f"Could not read PDF: {first_exc}") from first_exc

        # Pass 2 — pypdfium2 with empty password if the caller hadn't tried one.
        # This is the cheapest opener for owner-only-locked PDFs.
        if password is None:
            try:
                doc = _open_with_pypdfium(pdf_input, "")
            except pdfium.PdfiumError as second_exc:
                if not _looks_like_password_error(second_exc):
                    raise InvalidPdfError(f"Could not read PDF: {second_exc}") from second_exc
                doc = None  # fall through

        # Pass 3 — pikepdf fallback. Opens more encryption variants, and a
        # successful save() drops any encryption from the output stream.
        if doc is None:
            doc = _open_with_pikepdf_fallback(_as_bytes(pdf_input), password)

    try:
        chunks: list[TextChunk] = []
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_height = page.get_height()
            textpage = page.get_textpage()
            try:
                chunks.extend(_chunks_for_page(textpage, page_index + 1, page_height))
            finally:
                textpage.close()
                page.close()
        # Stable order: page, then top-to-bottom, then left-to-right.
        chunks.sort(key=lambda c: (c.page, c.y0, c.x0))
        return chunks
    finally:
        doc.close()


def _chunks_for_page(
    textpage: "pdfium.PdfTextPage",
    page_number: int,
    page_height: float,
) -> Iterable[TextChunk]:
    n = textpage.count_rects()
    for i in range(n):
        left, bottom, right, top = textpage.get_rect(i)
        text = textpage.get_text_bounded(left, bottom, right, top).strip()
        if not text:
            continue
        # Convert bottom-up → top-down: y grows downward, y0 is the top edge.
        y0 = page_height - top
        y1 = page_height - bottom
        yield TextChunk(
            text=text,
            x0=float(left),
            y0=float(y0),
            x1=float(right),
            y1=float(y1),
            page=page_number,
        )
