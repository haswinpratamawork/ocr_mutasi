"""Thin wrapper around pypdfium2 that yields positioned text chunks.

PDFium uses a bottom-left origin (Y grows upward). We convert to a top-left
origin (Y grows downward) here so every consumer downstream can reason about
"top of page" vs "bottom of page" naturally.

This module is the only place that imports pypdfium2. Failures from the
underlying library are re-raised as `InvalidPdfError` so callers never need
to import a third-party exception type to handle them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import pypdfium2 as pdfium

from .models import TextChunk

PdfInput = Union[str, Path, bytes]


class InvalidPdfError(ValueError):
    """The input bytes don't form a readable PDF (corrupt, truncated, not a PDF)."""


def extract_chunks(pdf_input: PdfInput) -> list[TextChunk]:
    """Open a PDF and return every text rect on every page as a TextChunk.

    Each rect is one contiguous run of text as identified by PDFium's text
    page (roughly: a span of characters with the same style sitting on the
    same baseline). This granularity matches what we need for column-based
    table reconstruction.

    Raises:
        InvalidPdfError: when the input isn't a readable PDF.
    """
    try:
        doc = pdfium.PdfDocument(pdf_input)
    except pdfium.PdfiumError as exc:
        raise InvalidPdfError(f"Could not read PDF: {exc}") from exc
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
