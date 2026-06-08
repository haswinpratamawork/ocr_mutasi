"""Orchestration: OCR a document, then classify its text.

`run` handles a single document and lets OCR errors propagate (the API maps
them to status codes). `run_batch` classifies many documents concurrently and
isolates per-file failures so one bad document never sinks the batch.
"""
from __future__ import annotations

import asyncio
import logging

from . import llm_classifier, ocr_client
from .config import get_settings
from .models import ClassificationResult, Confidence, DocumentType, OcrAudit

logger = logging.getLogger(__name__)


def _result(
    filename: str,
    doc_type: DocumentType,
    confidence: Confidence,
    reasoning: str,
    audit: OcrAudit,
    text: str,
    include_text: bool,
) -> ClassificationResult:
    return ClassificationResult(
        filename=filename,
        document_type=doc_type,
        confidence=confidence,
        reasoning=reasoning,
        audit=audit,
        text=text if include_text else None,
    )


async def run(file_bytes: bytes, filename: str, include_text: bool = False) -> ClassificationResult:
    """Classify a single document. Raises ocr_client.OcrError on OCR failure."""
    text, audit = await ocr_client.extract_text(file_bytes, filename)

    if not text.strip():
        # Nothing to classify — short-circuit, skip the LLM call.
        audit.errors.append("no text extracted from OCR")
        return _result(filename, DocumentType.unknown, Confidence.low, "", audit, text, include_text)

    # The Azure SDK client is synchronous; run it off the event loop so batch
    # concurrency isn't serialized behind it.
    doc_type, confidence, reasoning, err = await asyncio.to_thread(llm_classifier.classify, text)
    if err:
        audit.errors.append(err)
    return _result(filename, doc_type, confidence, reasoning, audit, text, include_text)


async def _run_capturing(
    file_bytes: bytes,
    filename: str,
    include_text: bool,
    sem: asyncio.Semaphore,
) -> ClassificationResult:
    """run() wrapper for batch use: turn any failure into a result so the
    batch as a whole still succeeds."""
    async with sem:
        try:
            return await run(file_bytes, filename, include_text)
        except ocr_client.OcrError as exc:
            return _result(
                filename,
                DocumentType.unknown,
                Confidence.low,
                "",
                OcrAudit(errors=[str(exc)]),
                "",
                include_text,
            )
        except Exception as exc:  # genuine bug — keep going, record it
            logger.exception("unexpected classify failure for %s", filename)
            return _result(
                filename,
                DocumentType.unknown,
                Confidence.low,
                "",
                OcrAudit(errors=[f"unexpected error: {exc}"]),
                "",
                include_text,
            )


async def run_batch(
    files: list[tuple[str, bytes]],
    include_text: bool = False,
) -> list[ClassificationResult]:
    """Classify many documents concurrently (capped), one result per file,
    in input order."""
    settings = get_settings()
    sem = asyncio.Semaphore(settings.batch_ocr_concurrency)
    tasks = [
        _run_capturing(data, filename, include_text, sem) for filename, data in files
    ]
    return await asyncio.gather(*tasks)
