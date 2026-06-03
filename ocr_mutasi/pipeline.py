"""End-to-end orchestration: PDF bytes in, ExtractionResponse out.

The pipeline detects the bank from page-1 text and dispatches to the matching
parser module under ``ocr_mutasi.parsers``. Everything downstream (credit
filtering, LLM classification, response shaping) is bank-agnostic.

`run_batch` accepts multiple PDFs in one call and feeds ALL their credits
into a single cross-month LLM classification — necessary to detect recurring
payroll deposits whose individual rows look unremarkable.
"""
from __future__ import annotations

from .llm_classifier import classify_credits, classify_credits_batch
from .models import (
    Audit,
    BatchAudit,
    BatchClassifiedCredit,
    BatchExtractionResponse,
    CategoryTotal,
    ClassifiedCredit,
    ExtractionResponse,
    FileExtraction,
)
from .parsers import SUPPORTED_BANKS, detect_bank, get_parser
from .pdf_extractor import extract_chunks


class UnsupportedBankError(ValueError):
    """Raised when the PDF doesn't match any known bank layout."""


def run(
    pdf_bytes: bytes,
    *,
    classify: bool = True,
    password: str | None = None,
) -> ExtractionResponse:
    """Run extraction + (optionally) LLM classification on a PDF.

    ``password`` is forwarded to the PDF opener so encrypted statements
    (Indonesian e-statements commonly use account-no / DOB / NIK) open
    cleanly when the caller provides the right value.
    """
    chunks = extract_chunks(pdf_bytes, password=password)
    bank = detect_bank(chunks)
    if bank == "UNKNOWN":
        raise UnsupportedBankError(
            "PDF doesn't match any known bank layout (supported: BCA Rekening Tahapan, BRI BritAma, Mandiri Tabungan, Permata Rekening Koran, Sinarmas Tabungan)."
        )
    parser = get_parser(bank)

    account = parser.parse_header(chunks)
    parsed = parser.parse_transactions(chunks, account)

    credits_only = [t for t in parsed.transactions if t.type == "CR"]
    if classify and credits_only:
        classified, classifier_err = classify_credits(credits_only)
        classifier_errors = [classifier_err] if classifier_err else []
    else:
        classified = [ClassifiedCredit(**t.model_dump()) for t in credits_only]
        classifier_errors = []

    pages = max((c.page for c in chunks), default=0)
    return ExtractionResponse(
        account=account,
        transactions=parsed.transactions,
        credits=classified,
        audit=Audit(
            pages_processed=pages,
            rows_detected=len(parsed.transactions),
            credit_count=len(credits_only),
            debit_count=sum(1 for t in parsed.transactions if t.type == "DB"),
            balance_warnings=parsed.balance_warnings,
            parse_warnings=parsed.parse_warnings,
            classifier_errors=classifier_errors,
        ),
    )


def _extract_one(
    filename: str,
    pdf_bytes: bytes,
    password: str | None = None,
) -> FileExtraction:
    """Extract a single file without classifying — used by run_batch."""
    chunks = extract_chunks(pdf_bytes, password=password)
    bank = detect_bank(chunks)
    if bank == "UNKNOWN":
        raise UnsupportedBankError(
            f"{filename}: PDF doesn't match any known bank layout "
            "(supported: BCA Rekening Tahapan, BRI BritAma, Mandiri Tabungan, Permata Rekening Koran, Sinarmas Tabungan)."
        )
    parser = get_parser(bank)
    account = parser.parse_header(chunks)
    parsed = parser.parse_transactions(chunks, account)
    pages = max((c.page for c in chunks), default=0)
    return FileExtraction(
        filename=filename,
        account=account,
        transactions=parsed.transactions,
        audit=Audit(
            pages_processed=pages,
            rows_detected=len(parsed.transactions),
            credit_count=sum(1 for t in parsed.transactions if t.type == "CR"),
            debit_count=sum(1 for t in parsed.transactions if t.type == "DB"),
            balance_warnings=parsed.balance_warnings,
            parse_warnings=parsed.parse_warnings,
        ),
    )


def run_batch(
    files: list[tuple[str, bytes]],
    *,
    classify: bool = True,
    password: str | None = None,
) -> BatchExtractionResponse:
    """Extract multiple PDFs and run a SINGLE cross-month classification.

    Each input is (filename, pdf_bytes). Per-file extraction is independent;
    only the LLM call sees every credit at once. The order of `files` is
    preserved in the response, and each returned credit carries `source_file`.
    """
    file_results: list[FileExtraction] = []
    credits_with_source: list[tuple[str, "Transaction"]] = []  # type: ignore[name-defined]
    for filename, data in files:
        fe = _extract_one(filename, data, password=password)
        file_results.append(fe)
        for tx in fe.transactions:
            if tx.type == "CR":
                credits_with_source.append((filename, tx))

    if classify and credits_with_source:
        classified, err = classify_credits_batch(credits_with_source)
        classifier_errors = [err] if err else []
    else:
        from .models import ClassifiedCredit as _CC
        classified = [_CC(**tx.model_dump()) for _, tx in credits_with_source]
        classifier_errors = []

    # Re-attach source_file to each classified credit (the classifier returns
    # them in the same order as the input list, so zip is sufficient).
    classified_with_src = [
        BatchClassifiedCredit(**cc.model_dump(), source_file=src)
        for cc, (src, _) in zip(classified, credits_with_source)
    ]

    # Aggregate per-category totals across the batch. `min` is the smallest
    # single-transaction amount in the category (None when the category is
    # empty) — useful for sanity-checking salary floors, smallest bonus, etc.
    cat_totals: dict[str, CategoryTotal] = {}
    for cat in ("Gaji", "THR", "Bonus", "Insentif", "Lainnya"):
        items = [c for c in classified_with_src if (c.category or "Lainnya") == cat]
        amounts = [c.amount for c in items]
        cat_totals[cat] = CategoryTotal(
            count=len(items),
            sum=sum(amounts),
            min=min(amounts) if amounts else None,
        )

    return BatchExtractionResponse(
        files=file_results,
        credits=classified_with_src,
        audit=BatchAudit(
            files_processed=len(file_results),
            transactions_total=sum(len(fe.transactions) for fe in file_results),
            credits_total=len(credits_with_source),
            classifier_errors=classifier_errors,
            category_totals=cat_totals,
        ),
    )
