#!/usr/bin/env python3
"""Single upload parser for Surat Keterangan Kerja PDFs."""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .extract_parser import (
    EmploymentCertificateAnalyzer,
    ParserConfig,
    PdfPasswordError,
    PdfTextExtractor,
    iter_pdfs,
    keyword_in,
    summary_to_jsonable,
    text_quality_needs_ocr,
    write_json,
)


BASE_DIR = Path(__file__).resolve().parent
REQUIRED_FIELDS = ("worker_name", "institution_name", "position")
SKK_NEGATIVE_KEYWORDS = (
    "slip gaji",
    "payslip",
    "invoice",
    "faktur",
    "rekening koran",
    "curriculum vitae",
    "ijazah",
    "kartu keluarga",
)
LLM_FALLBACK_ENV_NAMES = (
    "KETERANGAN_KERJA_FALLBACK",
    "SKK_LLM_FALLBACK",
    "LLM_FALLBACK",
    "SALARY_SLIP_FALLBACK",
)


class NonEmploymentCertificateError(RuntimeError):
    """Raised when a PDF does not look like a Surat Keterangan Kerja."""

    def __init__(self, source_file: str, classification: dict[str, Any]) -> None:
        super().__init__("Document is not recognized as a Surat Keterangan Kerja.")
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
            raise PdfPasswordError(f"{pdf_path.name} is password protected or the password is incorrect.")
        if not sys.stdin.isatty():
            raise PdfPasswordError(
                f"{pdf_path.name} is password protected. Run the script from an interactive terminal "
                "so it can ask for the PDF password."
            )

        for attempt in range(1, self.max_attempts + 1):
            try:
                password = getpass.getpass(
                    f"Password required for {pdf_path.name} (attempt {attempt}/{self.max_attempts}): "
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


def extracted_text_and_lines(extracted: dict[str, Any]) -> tuple[str, list[str]]:
    lines = [line for page in extracted.get("pages", []) for line in page.get("lines", [])]
    text = "\n".join(page.get("text", "") for page in extracted.get("pages", []))
    return text, lines


def classify_employment_certificate(extracted: dict[str, Any]) -> dict[str, Any]:
    config = ParserConfig()
    text, lines = extracted_text_and_lines(extracted)
    lowered = text.casefold()
    first_lines = "\n".join(lines[:10]).casefold()
    score = 0
    reasons: list[str] = []

    def add(points: int, reason: str) -> None:
        nonlocal score
        score += points
        reasons.append(reason)

    if keyword_in(first_lines, config.title_keywords) or keyword_in(lowered, config.title_keywords):
        add(35, "Found Surat Keterangan Kerja title keyword.")
    if keyword_in(lowered, config.worker_name_keywords):
        add(15, "Found worker/name keyword.")
    if keyword_in(lowered, config.position_keywords):
        add(15, "Found position keyword.")
    if keyword_in(lowered, config.start_date_keywords + config.status_keywords):
        add(15, "Found employment period/status keyword.")
    if keyword_in(lowered, config.institution_keywords):
        add(10, "Found institution keyword.")
    if keyword_in(lowered, config.signer_keywords):
        add(10, "Found signer/approval keyword.")

    negative_matches = [keyword for keyword in SKK_NEGATIVE_KEYWORDS if keyword in lowered]
    if negative_matches:
        score -= 30
        reasons.append("Found non-SKK document keyword: " + ", ".join(negative_matches[:3]) + ".")

    clamped_score = max(0, min(score, 100))
    if clamped_score >= 55:
        document_type = "surat_keterangan_kerja"
        status = "accepted"
    elif clamped_score >= 35:
        document_type = "possible_surat_keterangan_kerja"
        status = "uncertain"
    else:
        document_type = "unknown"
        status = "rejected"
        if not reasons:
            reasons.append("Missing Surat Keterangan Kerja signals.")

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


def compact_from_rule_summary(
    summary: dict[str, Any],
    extracted: dict[str, Any],
    extraction_method: str,
    source_file: str,
) -> dict[str, Any]:
    return {
        "source_file": source_file,
        "page_number": 1 if extracted.get("page_count") == 1 else None,
        "document_number": summary.get("document_number"),
        "document_date": summary.get("document_date"),
        "worker_name": summary.get("worker_name"),
        "employee_id": summary.get("employee_id"),
        "position": summary.get("position"),
        "department": summary.get("department"),
        "employment_status": summary.get("employment_status"),
        "start_date": summary.get("start_date"),
        "end_date": summary.get("end_date"),
        "tenure": summary.get("tenure"),
        "institution_name": summary.get("institution"),
        "signer_name": summary.get("signer_name"),
        "signer_position": summary.get("signer_position"),
        "purpose": summary.get("purpose"),
        "confidence_notes": summary.get("confidence_notes", []),
        "extraction_method": extraction_method,
    }


def document_to_indonesian(document: dict[str, Any]) -> dict[str, Any]:
    result = {
        "sumber_file": document.get("source_file"),
        "nomor_halaman": document.get("page_number"),
        "nomor_surat": document.get("document_number"),
        "tanggal_surat": document.get("document_date"),
        "nama_pekerja": document.get("worker_name"),
        "nik": document.get("employee_id"),
        "jabatan": document.get("position"),
        "departemen": document.get("department"),
        "status_karyawan": document.get("employment_status"),
        "tanggal_mulai_kerja": document.get("start_date"),
        "tanggal_akhir_kerja": document.get("end_date"),
        "masa_kerja": document.get("tenure"),
        "nama_institusi": document.get("institution_name"),
        "penandatangan": document.get("signer_name"),
        "jabatan_penandatangan": document.get("signer_position"),
        "tujuan_keterangan": document.get("purpose"),
        "metode_ekstraksi": document.get("extraction_method"),
    }
    notes = document.get("confidence_notes") or []
    if notes:
        result["catatan"] = notes
    classification = document.get("classification")
    if classification:
        result["classifikasi"] = classification
    return result


def unique_values(values: list[Any]) -> list[Any]:
    result = []
    seen = set()
    for value in values:
        if value in (None, "") or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def build_request_summary(
    documents: list[dict[str, Any]],
    output_path: Path,
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "metadata": {
            "dibuat_pada": datetime.now(timezone.utc).isoformat(),
            "jumlah_dokumen": len(documents),
            "folder_output": str(output_path),
        },
        "pemohon": {
            "nama_pekerja": unique_values([document.get("worker_name") for document in documents]),
            "nik": unique_values([document.get("employee_id") for document in documents]),
            "nama_institusi": unique_values([document.get("institution_name") for document in documents]),
        },
        "dokumen": [document_to_indonesian(document) for document in documents],
        "kesalahan": errors or [],
    }


def missing_required(document: dict[str, Any]) -> bool:
    return any(not document.get(field) for field in REQUIRED_FIELDS)


def env_flag_enabled(*names: str) -> bool:
    enabled_values = {"1", "true", "yes", "on", "always", "force", "enabled", "llm", "azure"}
    disabled_values = {"", "0", "false", "no", "off", "disabled", "none"}
    for name in names:
        value = os.environ.get(name, "").strip().casefold()
        if value in enabled_values:
            return True
        if value not in disabled_values:
            return True
    return False


def llm_fallback_enabled() -> bool:
    try:
        from .extract_llm import load_env_file

        load_env_file(BASE_DIR.parent / ".env")
    except Exception:
        pass
    return env_flag_enabled(*LLM_FALLBACK_ENV_NAMES)


def parse_with_llm_text(
    extracted: dict[str, Any],
    original_name: str,
    existing_document: dict[str, Any],
) -> dict[str, Any] | None:
    from .extract_llm import LLMTextMatcher

    matcher = LLMTextMatcher()
    for page in extracted.get("pages", []):
        page_text = page.get("text", "")
        if not page_text:
            continue
        document = matcher.match_page(
            page_text,
            original_name,
            page.get("page_number", 1),
            existing=existing_document,
        )
        document["confidence_notes"] = document.get("confidence_notes", []) + [
            "LLM text fallback used for SKK field matching."
        ]
        return document
    return None


def parse_pdf(pdf_path: Path, password: str | None = None) -> dict[str, Any]:
    config = ParserConfig()
    extractor = PdfTextExtractor()
    extracted = extractor.extract(pdf_path, password=password)
    used_extraction = extracted

    if text_quality_needs_ocr(extracted, config):
        try:
            from .extract_ocr import TesseractOcrExtractor

            used_extraction = TesseractOcrExtractor().extract(pdf_path, password=password)
        except RuntimeError as exc:
            extracted.setdefault("warnings", []).append(str(exc))
            used_extraction = extracted

    classification = classify_employment_certificate(used_extraction)
    if classification["status"] == "rejected":
        raise NonEmploymentCertificateError(pdf_path.name, classification)

    analyzer = EmploymentCertificateAnalyzer(config)
    rule_summary = summary_to_jsonable(analyzer.analyze(used_extraction))
    document = compact_from_rule_summary(
        rule_summary,
        used_extraction,
        used_extraction.get("extraction_method", "pdf_text"),
        pdf_path.name,
    )

    if missing_required(document) or llm_fallback_enabled():
        try:
            llm_document = parse_with_llm_text(used_extraction, pdf_path.name, document)
            if llm_document:
                document = {**document, **llm_document}
        except Exception as exc:  # noqa: BLE001 - keep LLM failures as parser notes.
            document.setdefault("confidence_notes", []).append(str(exc))

    result = {
        "source_file": str(pdf_path),
        "extracted": used_extraction,
        "rule_summary": rule_summary,
        "documents": [document],
    }
    return attach_classification(result, classification)


def parse_upload(
    input_path: str | Path,
    output_path: str | Path,
    password: str | None = None,
    allow_password_prompt: bool = True,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    provider = PdfPasswordProvider(initial_password=password, allow_prompt=allow_password_prompt)
    all_extracted = []
    documents = []
    errors: list[dict[str, Any]] = []

    for pdf_path in iter_pdfs(input_path):
        try:
            result = provider.parse_with_prompt(pdf_path)
        except NonEmploymentCertificateError as exc:
            errors.append(
                {
                    "sumber_file": exc.source_file,
                    "kesalahan": str(exc),
                    "classifikasi": exc.classification,
                }
            )
            continue
        except Exception as exc:  # noqa: BLE001 - keep batch processing best-effort.
            errors.append({"sumber_file": pdf_path.name, "kesalahan": str(exc)})
            continue
        all_extracted.append(result)
        documents.extend(result["documents"])

    summary = build_request_summary(documents, output_path, errors)
    extracted_payload = {
        "metadata": {
            "dibuat_pada": datetime.now(timezone.utc).isoformat(),
            "jumlah_dokumen": len(documents),
        },
        "results": all_extracted,
    }
    write_json(output_path / "extracted.json", extracted_payload)
    write_json(output_path / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Surat Keterangan Kerja fields from PDFs.")
    parser.add_argument("--input", required=True, help="PDF file or folder of PDFs.")
    parser.add_argument("--output", default=str(BASE_DIR / "output"), help="Output folder.")
    parser.add_argument("--password", default=None, help="Optional PDF password.")
    args = parser.parse_args()

    summary = parse_upload(args.input, args.output, password=args.password)
    print(f"Wrote {Path(args.output) / 'extracted.json'}")
    print(f"Wrote {Path(args.output) / 'summary.json'}")
    if summary.get("kesalahan"):
        print(f"Completed with {len(summary['kesalahan'])} error(s).")


if __name__ == "__main__":
    main()
