#!/usr/bin/env python3
"""Single upload parser for salary-slip PDFs.

Flow:
1. Parse the PDF text layer with pypdfium2 rules from extract_parser.py.
2. If required values are missing, extract text with local Tesseract OCR.
3. If required values or semantic keywords are still missing, use LLM
   text-only matching to map similar labels into fixed fields.
4. Write one extracted JSON and one summary JSON per PDF.
"""

from __future__ import annotations

import argparse
import getpass
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .extract_parser import (
    PdfPasswordError,
    PdfTextExtractor,
    ParserConfig,
    SalarySlipAnalyzer,
    keyword_in,
    iter_pdfs,
    money_matches,
    normalize_space,
    summary_to_jsonable,
    text_quality_needs_ocr,
    write_json,
)


BASE_DIR = Path(__file__).resolve().parent


BASE_SALARY_KEYWORDS = (
    "gaji",
    "gaji pokok",
    "upah pokok",
    "basic salary",
    "base salary",
    "base compensation",
    "imbalan dasar",
    "pokok",
)
DATE_RE = re.compile(
    r"\b("
    r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}"
    r"|"
    r"\d{4}[./-]\d{1,2}(?:[./-]\d{1,2})?"
    r")\b"
)
INDONESIAN_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+"
    r"(januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)"
    r"\s+(\d{4})\b",
    re.IGNORECASE,
)
INDONESIAN_MONTHS = {
    "januari": 1,
    "februari": 2,
    "maret": 3,
    "april": 4,
    "mei": 5,
    "juni": 6,
    "juli": 7,
    "agustus": 8,
    "september": 9,
    "oktober": 10,
    "november": 11,
    "desember": 12,
}
ENGLISH_DATE_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+(\d{4})\b",
    re.IGNORECASE,
)
ENGLISH_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
PERIOD_KEYWORDS = (
    "periode",
    "periode pembayaran",
    "pay period",
    "payment period",
    "pay date",
    "tanggal pembayaran",
)

REQUIRED_FIELDS = ("worker_name", "institution_name", "total_paid", "pokok", "deduction")
PAYROLL_KEYWORDS = (
    "gaji",
    "upah",
    "pokok",
    "penerimaan",
    "penghasilan",
    "pendapatan",
    "tunjangan",
    "thr",
    "bonus",
    "insentif",
    "allowance",
    "deduction",
    "potongan",
    "pengurangan",
    "pajak",
    "pph",
    "diterima",
    "take home",
    "net pay",
)
SALARY_TITLE_KEYWORDS = (
    "slip gaji",
    "slip upah",
    "pay slip",
    "payslip",
    "salary slip",
    "payroll statement",
)
SALARY_BASE_KEYWORDS = (
    "gaji pokok",
    "basic salary",
    "base salary",
    "upah pokok",
    "fixed compensation",
    "kompensasi tetap",
)
SALARY_TOTAL_KEYWORDS = (
    "take home pay",
    "total diterima",
    "total penerimaan",
    "jumlah di transfer",
    "bank transfer",
    "net pay",
    "total dibayar",
)
SALARY_DEDUCTION_KEYWORDS = (
    "potongan",
    "deduction",
    "pengurang",
    "pengurangan",
    "pajak",
    "pph",
    "bpjs",
    "jaminan pensiun",
)
SALARY_PERSON_PERIOD_KEYWORDS = (
    "employee",
    "karyawan",
    "nama",
    "period",
    "periode",
    "pay date",
    "tanggal pembayaran",
)
NON_SALARY_KEYWORDS = (
    "invoice",
    "faktur",
    "kwitansi",
    "receipt",
    "bank statement",
    "rekening koran",
    "surat keterangan",
    "perjanjian",
    "contract",
    "curriculum vitae",
    "resume",
    "ktp",
    "kartu keluarga",
    "ijazah",
)


class NonSalarySlipError(RuntimeError):
    """Raised when a PDF does not look like a salary slip."""

    def __init__(self, source_file: str, classification: dict[str, Any]) -> None:
        super().__init__("Document is not recognized as a salary slip.")
        self.source_file = source_file
        self.classification = classification


class PdfPasswordProvider:
    """Prompt for protected PDFs and reuse successful passwords during one run."""

    def __init__(
        self,
        initial_password: str | None = None,
        max_attempts: int = 3,
        allow_prompt: bool = True,
    ) -> None:
        self.max_attempts = max_attempts
        self.allow_prompt = allow_prompt
        self._known_passwords: list[str | None] = [None]
        if initial_password:
            self._known_passwords.append(initial_password)

    def parse_with_prompt(self, pdf_path: Path) -> dict[str, Any]:
        for password in self._known_passwords:
            try:
                return parse_pdf(pdf_path, password=password)
            except PdfPasswordError:
                continue

        if not self.allow_prompt:
            raise PdfPasswordError(
                f"{pdf_path.name} is password protected or the password is incorrect."
            )

        if not sys.stdin.isatty():
            raise PdfPasswordError(
                f"{pdf_path.name} is password protected. Run the script from an interactive terminal "
                "so it can ask for the PDF password."
            )

        for attempt in range(1, self.max_attempts + 1):
            try:
                password = getpass.getpass(
                    f"Password required for {pdf_path.name} "
                    f"(attempt {attempt}/{self.max_attempts}): "
                )
            except (EOFError, KeyboardInterrupt) as exc:
                raise PdfPasswordError(f"Password input cancelled for {pdf_path.name}.") from exc
            if not password:
                continue
            try:
                result = parse_pdf(pdf_path, password=password)
            except PdfPasswordError:
                print(f"Incorrect password for {pdf_path.name}.")
                continue
            self._known_passwords.append(password)
            return result

        raise PdfPasswordError(f"Could not open {pdf_path.name}: password required or incorrect.")


def money_or_zero(value: Any) -> int | float:
    return value if isinstance(value, (int, float)) else 0


def extracted_text_and_lines(extracted: dict[str, Any]) -> tuple[str, list[str]]:
    lines = [line for page in extracted.get("pages", []) for line in page.get("lines", [])]
    text = "\n".join(page.get("text", "") for page in extracted.get("pages", []))
    return text, lines


def classify_salary_slip(extracted: dict[str, Any]) -> dict[str, Any]:
    text, lines = extracted_text_and_lines(extracted)
    lowered = text.casefold()
    first_lines = "\n".join(lines[:10]).casefold()
    money_count = sum(len(money_matches(line)) for line in lines)
    score = 0
    reasons: list[str] = []

    def add(points: int, reason: str) -> None:
        nonlocal score
        score += points
        reasons.append(reason)

    if keyword_in(first_lines, SALARY_TITLE_KEYWORDS) or keyword_in(lowered, SALARY_TITLE_KEYWORDS):
        add(30, "Found salary-slip title keyword.")
    if keyword_in(lowered, SALARY_BASE_KEYWORDS):
        add(20, "Found base salary keyword.")
    if keyword_in(lowered, SALARY_TOTAL_KEYWORDS):
        add(20, "Found total paid or take-home-pay keyword.")
    if keyword_in(lowered, SALARY_DEDUCTION_KEYWORDS):
        add(15, "Found deduction/tax/payroll deduction keyword.")
    if keyword_in(lowered, SALARY_PERSON_PERIOD_KEYWORDS):
        add(10, "Found employee or payroll period keyword.")
    if money_count >= 3:
        add(10, "Found multiple money values.")
    elif money_count:
        add(5, "Found money value.")

    negative_matches = [keyword for keyword in NON_SALARY_KEYWORDS if keyword in lowered]
    if negative_matches:
        score -= 30
        reasons.append("Found non-salary document keyword: " + ", ".join(negative_matches[:3]) + ".")

    clamped_score = max(0, min(score, 100))
    if clamped_score >= 60:
        document_type = "salary_slip"
        status = "accepted"
    elif clamped_score >= 40:
        document_type = "possible_salary_slip"
        status = "uncertain"
    else:
        document_type = "unknown"
        status = "rejected"
        if not reasons:
            reasons.append("Missing salary-slip payroll signals.")

    return {
        "jenis_dokumen": document_type,
        "status": status,
        "confidence": round(clamped_score / 100, 2),
        "score": clamped_score,
        "alasan": reasons,
    }


def attach_classification(result: dict[str, Any], classification: dict[str, Any]) -> dict[str, Any]:
    result["classification"] = classification
    for document in result.get("documents", []):
        document["classification"] = classification
    result.get("extracted", {}).setdefault("classification", classification)
    return result


def base_salary_total(summary: dict[str, Any]) -> int | float:
    return sum(
        item.get("amount") or 0
        for item in summary.get("earnings", [])
        if any(keyword in item.get("label", "").casefold() for keyword in BASE_SALARY_KEYWORDS)
    )


def normalize_date(raw: str) -> str | None:
    parts = re.split(r"[./-]", raw.strip())
    if len(parts) == 2 and len(parts[0]) == 4:
        year, month = int(parts[0]), int(parts[1])
        return f"{year:04d}-{month:02d}"
    if len(parts) != 3:
        return None

    if len(parts[0]) == 4:
        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
    else:
        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
        if year < 100:
            year += 2000
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def normalize_indonesian_date(day: str, month: str, year: str) -> str:
    return f"{int(year):04d}-{INDONESIAN_MONTHS[month.casefold()]:02d}-{int(day):02d}"


def normalize_english_month_date(month: str, year: str) -> str:
    return f"{int(year):04d}-{ENGLISH_MONTHS[month.casefold()]:02d}"


def find_extracted_date(extracted: dict[str, Any]) -> str | None:
    for page in extracted.get("pages", []):
        lines = page.get("lines", [])
        for index, line in enumerate(lines):
            lowered = line.casefold()
            if not any(keyword in lowered for keyword in PERIOD_KEYWORDS):
                continue
            nearby = [line]
            nearby.extend(lines[index + 1 : index + 4])
            for candidate in nearby:
                match = DATE_RE.search(candidate)
                if match:
                    return normalize_date(match.group(1))
                match = INDONESIAN_DATE_RE.search(candidate)
                if match:
                    return normalize_indonesian_date(*match.groups())
                match = ENGLISH_DATE_RE.search(candidate)
                if match:
                    return normalize_english_month_date(*match.groups())

    for page in extracted.get("pages", []):
        for line in page.get("lines", []):
            match = DATE_RE.search(line)
            if match:
                return normalize_date(match.group(1))
            match = INDONESIAN_DATE_RE.search(line)
            if match:
                return normalize_indonesian_date(*match.groups())
            match = ENGLISH_DATE_RE.search(line)
            if match:
                return normalize_english_month_date(*match.groups())
    return None


def date_parts(value: str | None) -> dict[str, int | None]:
    if not value:
        return {"hari": None, "bulan": None, "tahun": None}
    parts = value.split("-")
    if len(parts) == 2:
        return {"hari": None, "bulan": int(parts[1]), "tahun": int(parts[0])}
    if len(parts) == 3:
        return {"hari": int(parts[2]), "bulan": int(parts[1]), "tahun": int(parts[0])}
    return {"hari": None, "bulan": None, "tahun": None}


def compact_from_rule_summary(
    summary: dict[str, Any],
    extracted: dict[str, Any],
    extraction_method: str,
    source_file: str,
) -> dict[str, Any]:
    pokok = base_salary_total(summary)
    incentive = money_or_zero(summary.get("incentive_total"))
    deduction = money_or_zero(summary.get("deduction_total"))
    total_paid = summary.get("paid_salary_total")
    confidence_notes = list(summary.get("confidence_notes", []))

    if total_paid is None and (pokok or incentive or deduction):
        total_paid = pokok + incentive - deduction
        confidence_notes.append("total_paid inferred as pokok + incentive - deduction.")

    return {
        "source_file": source_file,
        "page_number": 1 if extracted.get("page_count") == 1 else None,
        "extracted_date": find_extracted_date(extracted),
        "worker_name": summary.get("worker_name"),
        "institution_name": summary.get("institution"),
        "total_paid": total_paid,
        "pokok": pokok,
        "incentive": incentive,
        "deduction": deduction,
        "confidence_notes": confidence_notes,
        "extraction_method": extraction_method,
    }


def totals_for_documents(documents: list[dict[str, Any]]) -> dict[str, int | float]:
    return {
        "total_paid": sum(item.get("total_paid") or 0 for item in documents),
        "pokok": sum(item.get("pokok") or 0 for item in documents),
        "incentive": sum(item.get("incentive") or 0 for item in documents),
        "deduction": sum(item.get("deduction") or 0 for item in documents),
    }


def clean_worker_name(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    cleaned = normalize_space(value).strip(" ,;:")
    cleaned = re.sub(
        r"(?i)^(?:drg|dr|dra|drs|prof|ir|h|hj|mr|mrs|ms)\.?\s+",
        "",
        cleaned,
    )
    return normalize_space(cleaned).strip(" ,;:")


def clean_institution_name(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    cleaned = normalize_space(value).strip(" ,;:")
    match = re.search(r"(?i)\b(?:PT|CV)\.?\s+.+", cleaned)
    if match:
        cleaned = cleaned[match.start() :]
    return normalize_space(cleaned).strip(" ,;:")


def normalize_document_identity(document: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(document)
    normalized["worker_name"] = clean_worker_name(normalized.get("worker_name"))
    normalized["institution_name"] = clean_institution_name(normalized.get("institution_name"))
    return normalized


def document_to_indonesian(document: dict[str, Any]) -> dict[str, Any]:
    document = normalize_document_identity(document)
    result = {
        "sumber_file": document.get("source_file"),
        "nomor_halaman": document.get("page_number"),
        "tanggal_periode": document.get("extracted_date"),
        "nama_pekerja": document.get("worker_name"),
        "nama_institusi": document.get("institution_name"),
        "total_dibayar": document.get("total_paid"),
        "gaji_pokok": document.get("pokok"),
        "tunjangan": document.get("incentive"),
        "potongan": document.get("deduction"),
        "metode_ekstraksi": document.get("extraction_method"),
    }
    notes = document.get("confidence_notes") or []
    if notes:
        result["catatan"] = notes
    classification = document.get("classification")
    if classification:
        result["classifikasi"] = classification
    return result


def totals_to_indonesian(totals: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_dibayar": totals.get("total_paid", 0),
        "gaji_pokok": totals.get("pokok", 0),
        "tunjangan": totals.get("incentive", 0),
        "potongan": totals.get("deduction", 0),
    }


def unique_values(values: list[Any]) -> list[Any]:
    result = []
    seen = set()
    for value in values:
        if value in (None, "") or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def monthly_summaries(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], dict[str, Any]] = {}

    for document in documents:
        parts = date_parts(document.get("extracted_date"))
        month = parts.get("bulan")
        year = parts.get("tahun")
        if not month or not year:
            continue

        key = (year, month)
        if key not in grouped:
            grouped[key] = {
                "bulan": month,
                "tahun": year,
                "periode": f"{year:04d}-{month:02d}",
                "jumlah_dokumen": 0,
                "total": {
                    "total_paid": 0,
                    "pokok": 0,
                    "incentive": 0,
                    "deduction": 0,
                },
            }

        grouped[key]["jumlah_dokumen"] += 1
        for field in ("total_paid", "pokok", "incentive", "deduction"):
            grouped[key]["total"][field] += document.get(field) or 0

    summaries = []
    for item in grouped.values():
        summaries.append(
            {
                "bulan": item["bulan"],
                "tahun": item["tahun"],
                "periode": item["periode"],
                "jumlah_dokumen": item["jumlah_dokumen"],
                "total": totals_to_indonesian(item["total"]),
            }
        )
    return sorted(summaries, key=lambda item: (item["tahun"], item["bulan"]))


def average_monthly_money(months: list[dict[str, Any]], indonesian_field: str) -> int | float:
    values = [
        item.get("total", {}).get(indonesian_field)
        for item in months
        if isinstance(item.get("total", {}).get(indonesian_field), (int, float))
    ]
    if not values:
        return 0
    average = sum(values) / len(values)
    return int(average) if average.is_integer() else round(average, 2)


def average_totals(documents: list[dict[str, Any]], months: list[dict[str, Any]]) -> dict[str, int | float]:
    return {
        "total_dibayar": average_monthly_money(months, "total_dibayar")
        if months
        else average_document_money(documents, "total_paid"),
        "gaji_pokok": average_monthly_money(months, "gaji_pokok")
        if months
        else average_document_money(documents, "pokok"),
        "tunjangan": average_monthly_money(months, "tunjangan")
        if months
        else average_document_money(documents, "incentive"),
        "potongan": average_monthly_money(months, "potongan")
        if months
        else average_document_money(documents, "deduction"),
    }


def average_document_money(documents: list[dict[str, Any]], field: str) -> int | float:
    values = [document.get(field) for document in documents if isinstance(document.get(field), (int, float))]
    if not values:
        return 0
    average = sum(values) / len(values)
    return int(average) if average.is_integer() else round(average, 2)


def build_nasabah_summary(
    documents: list[dict[str, Any]],
    output_path: Path,
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    documents = [normalize_document_identity(document) for document in documents]
    totals = totals_for_documents(documents)
    months = monthly_summaries(documents)
    return {
        "metadata": {
            "dibuat_pada": datetime.now(timezone.utc).isoformat(),
            "jumlah_dokumen": len(documents),
            "folder_output": str(output_path),
        },
        "nasabah": {
            "nama_pekerja": unique_values([document.get("worker_name") for document in documents]),
            "nama_institusi": unique_values([document.get("institution_name") for document in documents]),
        },
        "periode": {
            "jumlah": len(months),
            "rata_rata": average_totals(documents, months),
            "total": totals_to_indonesian(totals),
            "bulanan": months,
        },
        "dokumen": [document_to_indonesian(document) for document in documents],
        "kesalahan": [error_to_indonesian(error) for error in errors or []],
    }


def build_combined_extracted(
    parsed_results: list[dict[str, Any]],
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    pages = []
    methods = []
    warnings = []
    classifications = []
    global_page_number = 1

    for parsed in parsed_results:
        extracted = parsed.get("extracted", {})
        source_file = extracted.get("source_file")
        classification = parsed.get("classification") or extracted.get("classification")
        if classification:
            classifications.append({"source_file": source_file, "classification": classification})
        method = extracted.get("extraction_method")
        if method and method not in methods:
            methods.append(method)
        warnings.extend(extracted.get("warnings", []))

        for page in extracted.get("pages", []):
            combined_page = dict(page)
            combined_page["source_file"] = source_file
            combined_page["global_page_number"] = global_page_number
            if classification:
                combined_page["classification"] = classification
            pages.append(combined_page)
            global_page_number += 1

    return {
        "source_file": "multiple_pdfs" if len(parsed_results) > 1 else (
            parsed_results[0].get("extracted", {}).get("source_file") if parsed_results else None
        ),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "extraction_method": "+".join(methods) if methods else None,
        "page_count": len(pages),
        "pages": pages,
        "classifications": classifications,
        "warnings": unique_values(warnings),
        "errors": errors or [],
    }


def clear_previous_output_json(output_path: Path) -> None:
    if not output_path.exists():
        return
    for json_path in output_path.glob("*.json"):
        json_path.unlink()


def error_to_indonesian(error: dict[str, Any]) -> dict[str, Any]:
    result = {
        "sumber_file": error.get("source_file"),
        "kesalahan": error.get("error"),
    }
    if error.get("classification"):
        result["classifikasi"] = error["classification"]
    return result


def missing_required_value(document: dict[str, Any]) -> bool:
    for field in REQUIRED_FIELDS:
        value = document.get(field)
        if value is None or value == "":
            return True
        if field in {"total_paid", "pokok"} and value == 0:
            return True
    return False


def text_has_payroll_keywords(extracted: dict[str, Any]) -> bool:
    lines = [line for page in extracted.get("pages", []) for line in page.get("lines", [])]
    return any(keyword_in(line, PAYROLL_KEYWORDS) for line in lines)


def should_use_ocr(
    extracted: dict[str, Any],
    document: dict[str, Any],
    config: ParserConfig,
) -> bool:
    if text_quality_needs_ocr(extracted, config):
        return True
    return missing_required_value(document)


def should_use_llm_text(extracted: dict[str, Any], documents: list[dict[str, Any]]) -> bool:
    if extracted.get("extraction_method") == "ocr_tesseract":
        return True
    if any(missing_required_value(document) for document in documents):
        return True
    return not text_has_payroll_keywords(extracted)


def parse_with_rules(
    pdf_path: Path,
    original_name: str,
    config: ParserConfig,
    password: str | None = None,
) -> dict[str, Any]:
    extractor = PdfTextExtractor()
    analyzer = SalarySlipAnalyzer(config)
    extracted = extractor.extract(pdf_path, password=password)
    extracted["source_file"] = original_name
    summary = summary_to_jsonable(analyzer.analyze(extracted))
    summary["source_file"] = original_name
    document = compact_from_rule_summary(
        summary,
        extracted,
        extracted.get("extraction_method", "pdf_text"),
        original_name,
    )
    return {"extracted": extracted, "summary": summary, "documents": [document]}


def parse_extracted_with_rules(extracted: dict[str, Any], original_name: str, config: ParserConfig) -> dict[str, Any]:
    summaries = []
    documents = []
    for page in extracted.get("pages", []):
        page_extracted = {
            **extracted,
            "page_count": 1,
            "pages": [page],
            "source_file": original_name,
        }
        analyzer = SalarySlipAnalyzer(config)
        summary = summary_to_jsonable(analyzer.analyze(page_extracted))
        summary["source_file"] = original_name
        document = compact_from_rule_summary(
            summary,
            page_extracted,
            extracted.get("extraction_method", "text"),
            original_name,
        )
        document["page_number"] = page["page_number"]
        if extracted.get("page_count", 1) > 1:
            document["source_file"] = f"{original_name}#page-{page['page_number']}"
        summaries.append(summary)
        documents.append(document)
    return {"extracted": extracted, "summary": summaries, "documents": documents}


def parse_with_local_ocr(
    pdf_path: Path,
    original_name: str,
    config: ParserConfig,
    password: str | None = None,
) -> dict[str, Any]:
    from .extract_ocr import TesseractOcrExtractor

    extracted = TesseractOcrExtractor().extract(pdf_path, password=password)
    return parse_extracted_with_rules(extracted, original_name, config)


def parse_with_llm_text(
    extracted: dict[str, Any],
    original_name: str,
    existing_documents: list[dict[str, Any]],
) -> dict[str, Any]:
    from .extract_llm import LLMTextMatcher

    matcher = LLMTextMatcher()
    documents = []
    page_payloads = []
    for page in extracted.get("pages", []):
        page_number = page["page_number"]
        existing = existing_documents[min(page_number - 1, len(existing_documents) - 1)] if existing_documents else {}
        document = matcher.match_page(
            page.get("text", ""),
            source_file=original_name,
            page_number=page_number,
            existing=existing,
        )
        documents.append(document)
        page_payloads.append(
            {
                "page_number": page_number,
                "text": page.get("text", ""),
                "lines": page.get("lines", []),
                "hasil_pencocokan_llm": document_to_indonesian(document),
            }
        )

    return {
        "extracted": {
            "source_file": original_name,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "extraction_method": "ocr_text_llm_match",
            "page_count": len(page_payloads),
            "pages": page_payloads,
            "warnings": ["LLM text-only matching fallback used."],
        },
        "summary": documents,
        "documents": documents,
    }


def parse_pdf(pdf_path: Path, password: str | None = None) -> dict[str, Any]:
    config = ParserConfig()
    original_name = pdf_path.name
    rule_result = parse_with_rules(pdf_path, original_name, config, password=password)
    rule_document = rule_result["documents"][0]
    rule_classification = classify_salary_slip(rule_result["extracted"])
    text_weak = text_quality_needs_ocr(rule_result["extracted"], config)
    needs_ocr = should_use_ocr(rule_result["extracted"], rule_document, config)

    if rule_classification["status"] == "rejected" and not text_weak:
        raise NonSalarySlipError(original_name, rule_classification)

    if not needs_ocr:
        return attach_classification(rule_result, rule_classification)

    try:
        ocr_result = parse_with_local_ocr(pdf_path, original_name, config, password=password)
    except PdfPasswordError:
        raise
    except Exception as exc:
        rule_document["confidence_notes"].append(f"Local OCR fallback failed: {exc}")
        if rule_classification["status"] == "rejected":
            raise NonSalarySlipError(original_name, rule_classification)
        return attach_classification(rule_result, rule_classification)

    ocr_classification = classify_salary_slip(ocr_result["extracted"])
    if ocr_classification["status"] == "rejected":
        if rule_classification["status"] == "rejected":
            raise NonSalarySlipError(original_name, ocr_classification)
        return attach_classification(rule_result, rule_classification)

    if not should_use_llm_text(ocr_result["extracted"], ocr_result["documents"]):
        return attach_classification(ocr_result, ocr_classification)

    try:
        llm_result = parse_with_llm_text(ocr_result["extracted"], original_name, ocr_result["documents"])
        return attach_classification(llm_result, ocr_classification)
    except Exception as exc:
        for document in ocr_result["documents"]:
            document["confidence_notes"].append(f"LLM text fallback failed: {exc}")
        return attach_classification(ocr_result, ocr_classification)


def parse_upload(
    input_path: Path,
    output_path: Path,
    password: str | None = None,
    allow_password_prompt: bool = True,
) -> dict[str, Any]:
    if not input_path.is_absolute():
        cwd_input_path = input_path
        script_input_path = BASE_DIR / input_path
        input_path = cwd_input_path if cwd_input_path.exists() else script_input_path

    if not output_path.is_absolute():
        output_path = BASE_DIR / output_path

    pdfs = iter_pdfs(input_path)
    if not pdfs:
        raise SystemExit(f"No PDF files found in {input_path}")

    documents = []
    parsed_results = []
    errors = []
    password_provider = PdfPasswordProvider(
        initial_password=password,
        allow_prompt=allow_password_prompt,
    )

    for pdf_path in pdfs:
        try:
            parsed = password_provider.parse_with_prompt(pdf_path)
        except NonSalarySlipError as exc:
            errors.append(
                {
                    "source_file": exc.source_file,
                    "error": str(exc),
                    "classification": exc.classification,
                }
            )
            continue
        except Exception as exc:
            errors.append({"source_file": pdf_path.name, "error": str(exc)})
            continue

        parsed_results.append(parsed)
        documents.extend(parsed["documents"])

    combined_extracted = build_combined_extracted(parsed_results, errors)
    combined_summary = build_nasabah_summary(documents, output_path, errors)
    clear_previous_output_json(output_path)
    write_json(output_path / "extracted.json", combined_extracted)
    write_json(output_path / "summary.json", combined_summary)
    return combined_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload one PDF or a folder of PDFs and extract salary JSON."
    )
    parser.add_argument("-i", "--input", default="input", type=Path, help="PDF file or folder.")
    parser.add_argument("-o", "--output", default="output", type=Path, help="Output JSON folder.")
    parser.add_argument("--password", help="Optional password for protected PDFs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate = parse_upload(args.input, args.output, password=args.password)
    metadata = aggregate["metadata"]
    print(f"Parsed {metadata['jumlah_dokumen']} salary result(s).")
    print(f"Output JSON folder: {Path(metadata['folder_output'])}")
    for document in aggregate["dokumen"]:
        for note in document.get("catatan", []):
            if "fallback failed" in note:
                print(f"Warning for {document.get('sumber_file')}: {note}")
    if aggregate["kesalahan"]:
        print(f"Completed with {len(aggregate['kesalahan'])} error(s).")


if __name__ == "__main__":
    main()
