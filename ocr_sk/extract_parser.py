#!/usr/bin/env python3
"""Extract and summarize Surat Keterangan Kerja PDFs."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pypdfium2 as pdfium


LABEL_PREFIX_RE = re.compile(r"^(?:[\s\-•*–—]+|\d+[.)]\s*)")
DATE_NUMERIC_RE = re.compile(r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}[./-]\d{1,2}[./-]\d{1,2})\b")
DATE_WORD_RE = re.compile(
    r"\b(\d{1,2})\s+"
    r"(januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)"
    r"\s+(\d{4})\b",
    re.IGNORECASE,
)
MONTHS = {
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


class PdfPasswordError(RuntimeError):
    """Raised when a PDF cannot be opened without a valid password."""


def open_pdf_document(pdf_path: Path, password: str | None = None) -> pdfium.PdfDocument:
    """Open a PDF and normalize password-related errors."""

    try:
        return pdfium.PdfDocument(pdf_path, password=password)
    except pdfium.PdfiumError as exc:
        message = str(exc)
        lowered = message.casefold()
        if any(keyword in lowered for keyword in ("password", "encrypted", "security", "protected")):
            raise PdfPasswordError(f"{pdf_path.name} is password protected or the password is incorrect.") from exc
        raise


@dataclass(frozen=True)
class ParserConfig:
    """Keywords used by the SKK inference layer."""

    title_keywords: tuple[str, ...] = (
        "surat keterangan kerja",
        "surat keterangan aktif kerja",
        "surat keterangan pengangkatan",
        "keterangan kerja",
        "keterangan pengangkatan",
        "pengangkatan terakhir",
        "employment certificate",
        "certificate of employment",
        "reference letter",
    )
    worker_name_keywords: tuple[str, ...] = ("nama", "name", "karyawan", "pegawai", "pekerja")
    id_keywords: tuple[str, ...] = ("nik", "nip", "employee id", "nomor induk", "no. karyawan")
    position_keywords: tuple[str, ...] = ("jabatan", "posisi", "position", "job title")
    department_keywords: tuple[str, ...] = ("departemen", "department", "divisi", "unit kerja", "bagian")
    status_keywords: tuple[str, ...] = ("status", "tetap", "kontrak", "permanent", "contract")
    start_date_keywords: tuple[str, ...] = ("mulai bekerja", "tanggal masuk", "bergabung", "sejak", "start date")
    end_date_keywords: tuple[str, ...] = ("sampai dengan", "berakhir", "tanggal akhir", "end date")
    institution_keywords: tuple[str, ...] = ("pt.", "pt ", "cv.", "cv ", "persero", "perusahaan", "company")
    signer_keywords: tuple[str, ...] = ("hormat kami", "ditandatangani", "mengetahui", "hrd", "direktur", "manager")
    purpose_keywords: tuple[str, ...] = (
        "digunakan untuk",
        "keperluan",
        "tujuan",
        "menerangkan bahwa",
        "menerangkan dengan sebenarnya",
    )

    @classmethod
    def from_json(cls, path: Path) -> "ParserConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"Unknown config field(s): {', '.join(unknown)}")
        return cls(**{key: tuple(value) for key, value in raw.items()})


@dataclass
class EmploymentCertificateSummary:
    source_file: str
    document_number: str | None = None
    document_date: str | None = None
    worker_name: str | None = None
    employee_id: str | None = None
    position: str | None = None
    department: str | None = None
    employment_status: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    tenure: str | None = None
    institution: str | None = None
    signer_name: str | None = None
    signer_position: str | None = None
    purpose: str | None = None
    confidence_notes: list[str] = field(default_factory=list)


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def keyword_in(text: str, keywords: Iterable[str]) -> bool:
    lowered = text.casefold()
    return any(keyword.casefold() in lowered for keyword in keywords)


def clean_label_value(value: str) -> str:
    value = normalize_space(value)
    value = LABEL_PREFIX_RE.sub("", value)
    return normalize_space(value.strip(" :;|-"))


def iter_pdfs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.casefold() != ".pdf":
            raise ValueError(f"{input_path} is not a PDF file.")
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    pdfs = sorted(path for path in input_path.iterdir() if path.suffix.casefold() == ".pdf")
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {input_path}")
    return pdfs


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def summary_to_jsonable(summary: EmploymentCertificateSummary) -> dict[str, Any]:
    return asdict(summary)


def text_quality_needs_ocr(extracted: dict[str, Any], config: ParserConfig | None = None) -> bool:
    config = config or ParserConfig()
    text = "\n".join(page["text"] for page in extracted["pages"])
    lines = [line for page in extracted["pages"] for line in page["lines"]]
    page_count = max(extracted["page_count"], 1)
    total_chars = len(text)
    avg_chars = total_chars / page_count
    has_keyword = keyword_in(text, config.title_keywords + config.worker_name_keywords + config.position_keywords)
    return total_chars < 150 or avg_chars < 80 or len(lines) < 5 or not has_keyword


def normalize_date(raw: str | None) -> str | None:
    if not raw:
        return None
    raw = normalize_space(raw)
    word_match = DATE_WORD_RE.search(raw)
    if word_match:
        day, month, year = word_match.groups()
        return f"{int(year):04d}-{MONTHS[month.casefold()]:02d}-{int(day):02d}"
    numeric_match = DATE_NUMERIC_RE.search(raw)
    if not numeric_match:
        return None
    parts = re.split(r"[./-]", numeric_match.group(1))
    try:
        if len(parts[0]) == 4:
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
        else:
            day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
            if year < 100:
                year += 2000
    except (ValueError, IndexError):
        return None
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def first_date(text: str) -> str | None:
    word = DATE_WORD_RE.search(text)
    if word:
        return normalize_date(word.group(0))
    numeric = DATE_NUMERIC_RE.search(text)
    if numeric:
        return normalize_date(numeric.group(1))
    return None


class PdfTextExtractor:
    """Extract raw text from PDFs with pypdfium2."""

    def extract(self, pdf_path: Path, password: str | None = None) -> dict[str, Any]:
        document = open_pdf_document(pdf_path, password=password)
        pages: list[dict[str, Any]] = []
        try:
            for page_index, page in enumerate(document):
                text_page = page.get_textpage()
                text = text_page.get_text_range()
                lines = [normalize_space(line) for line in text.splitlines() if normalize_space(line)]
                pages.append(
                    {
                        "page_number": page_index + 1,
                        "char_count": len(text),
                        "text": text,
                        "lines": lines,
                    }
                )
                text_page.close()
                page.close()
        finally:
            document.close()
        return {
            "source_file": str(pdf_path),
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "extraction_method": "pdf_text",
            "page_count": len(pages),
            "pages": pages,
            "warnings": [] if any(page["char_count"] for page in pages) else ["No text layer found."],
        }


class EmploymentCertificateAnalyzer:
    """Infer SKK fields from extracted text."""

    def __init__(self, config: ParserConfig | None = None) -> None:
        self.config = config or ParserConfig()

    def analyze(self, extracted: dict[str, Any]) -> EmploymentCertificateSummary:
        lines = self._flatten_lines(extracted)
        text = "\n".join(item["text"] for item in lines)
        summary = EmploymentCertificateSummary(source_file=extracted["source_file"])
        summary.document_number = self._find_document_number(lines)
        summary.document_date = self._find_document_date(lines, text)
        summary.worker_name = self._find_worker_name(lines, text)
        summary.employee_id = self._find_employee_id(lines)
        summary.position = self._find_position(lines, text)
        summary.department = self._find_by_labels(lines, self.config.department_keywords)
        summary.employment_status = self._find_status(lines, text)
        summary.start_date = self._find_date_near(lines, self.config.start_date_keywords)
        summary.end_date = self._find_date_near(lines, self.config.end_date_keywords)
        summary.tenure = self._find_tenure(lines, text)
        summary.institution = self._find_institution(lines)
        summary.signer_name, summary.signer_position = self._find_signer(lines)
        summary.purpose = self._find_purpose(lines)
        if extracted.get("warnings"):
            summary.confidence_notes.extend(extracted["warnings"])
        return summary

    def _flatten_lines(self, extracted: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {"page": page["page_number"], "line_number": index, "text": line}
            for page in extracted["pages"]
            for index, line in enumerate(page["lines"], start=1)
        ]

    def _label_pattern(self, labels: tuple[str, ...]) -> str:
        parts = []
        for label in labels:
            escaped = re.escape(label)
            if re.fullmatch(r"[\w ]+", label):
                parts.append(r"\b" + escaped.replace(r"\ ", r"\s+") + r"\b")
            else:
                parts.append(escaped)
        return r"(?i)(?:" + "|".join(parts) + r")"

    def _find_by_labels(self, lines: list[dict[str, Any]], labels: tuple[str, ...]) -> str | None:
        label_pattern = self._label_pattern(labels)
        for index, item in enumerate(lines):
            line = item["text"]
            if not re.search(label_pattern, line):
                continue
            pattern = label_pattern + r"\s*[:=\-]?\s*(.+)"
            match = re.search(pattern, line)
            if match:
                value = clean_label_value(match.group(1))
                if value and not keyword_in(value, labels):
                    return self._trim_inline_metadata(value)
            if index + 1 < len(lines):
                value = clean_label_value(lines[index + 1]["text"])
                if value and not keyword_in(value, labels):
                    return self._trim_inline_metadata(value)
        return None

    def _trim_inline_metadata(self, value: str) -> str:
        value = re.split(
            r"(?i)\s{2,}|\s+\b(?:nik|nip|jabatan|posisi|departemen|status|tanggal)\b\s*:?",
            value,
            maxsplit=1,
        )[0]
        return clean_label_value(value)

    def _find_worker_name(self, lines: list[dict[str, Any]], text: str) -> str | None:
        for index, item in enumerate(lines):
            if not keyword_in(item["text"], ("menerangkan bahwa", "bahwa", "menyatakan dengan")):
                continue
            nearby = lines[index : index + 8]
            value = self._find_by_labels(nearby, ("nama", "name"))
            if value and self._looks_like_person(value):
                return value
        labeled = self._find_by_labels(lines, self.config.worker_name_keywords)
        if labeled and self._looks_like_person(labeled):
            return labeled
        people = [clean_label_value(line["text"]) for line in lines if self._looks_like_person(line["text"])]
        return people[0] if people else None

    def _find_employee_id(self, lines: list[dict[str, Any]]) -> str | None:
        patterns = (
            r"(?i)\bNIK\b\s*[:=\-]?\s*([A-Z0-9./\-]+)",
            r"(?i)\bNIP\b\s*[:=\-]?\s*([A-Z0-9./\-]+)",
            r"(?i)\bNo\.?\s*KTP\s*[:=\-]?\s*([A-Z0-9./\-]+)",
            r"(?i)\b(?:employee id|nomor induk|no\. karyawan)\s*[:=\-]?\s*([A-Z0-9./\-]+)",
        )
        for item in lines:
            for pattern in patterns:
                match = re.search(pattern, item["text"])
                if match:
                    return clean_label_value(match.group(1))
        return None

    def _find_position(self, lines: list[dict[str, Any]], text: str) -> str | None:
        body_patterns = (
            r"(?i)\bposisi\s+sebagai\s+([^.;,\n]+)",
            r"(?i)\bjabatan\s+terkini\s+sebagai\s+([^.;,\n]+)",
            r"(?i)\bdengan\s+jabatan\s+([^.;,\n]+)",
        )
        for pattern in body_patterns:
            match = re.search(pattern, text)
            if match:
                return clean_label_value(match.group(1))
        return self._find_by_labels(lines, self.config.position_keywords)

    def _find_document_number(self, lines: list[dict[str, Any]]) -> str | None:
        patterns = (
            r"(?i)\b(?:nomor|number)\s*(?:surat)?\s*[:=\-]\s*([A-Z0-9][A-Z0-9./\- ]+)",
            r"(?i)^\s*No\.?\s*[:=\-]\s*([A-Z0-9][A-Z0-9./\- ]+)",
            r"(?i)^\s*No\.?\s+([A-Z0-9][A-Z0-9./\-]+(?:/[A-Z0-9./\-]+)+)",
        )
        for item in lines[:20]:
            for pattern in patterns:
                match = re.search(pattern, item["text"])
                if match:
                    return clean_label_value(match.group(1))
        return None

    def _find_document_date(self, lines: list[dict[str, Any]], text: str) -> str | None:
        for item in reversed(lines[-12:]):
            date = first_date(item["text"])
            if date:
                return date
        return first_date(text)

    def _find_date_near(self, lines: list[dict[str, Any]], keywords: tuple[str, ...]) -> str | None:
        for index, item in enumerate(lines):
            if not keyword_in(item["text"], keywords):
                continue
            window = " ".join(line["text"] for line in lines[index : index + 3])
            date = first_date(window)
            if date:
                return date
        return None

    def _find_status(self, lines: list[dict[str, Any]], text: str) -> str | None:
        labeled = self._find_by_labels(lines, self.config.status_keywords)
        if labeled:
            return labeled
        lowered = text.casefold()
        if "karyawan tetap" in lowered or "pegawai tetap" in lowered or "permanent" in lowered:
            return "Tetap"
        if "karyawan kontrak" in lowered or "pegawai kontrak" in lowered or "contract" in lowered:
            return "Kontrak"
        if "magang" in lowered or "intern" in lowered:
            return "Magang"
        return None

    def _find_tenure(self, lines: list[dict[str, Any]], text: str) -> str | None:
        pattern = re.compile(r"(?i)\b(?:masa kerja|telah bekerja selama)\s*[:=\-]?\s*([^.;,\n]+)")
        match = pattern.search(text)
        if match:
            return clean_label_value(match.group(1))
        for line in lines:
            match = re.search(r"(?i)\b(\d+\s+tahun(?:\s+\d+\s+bulan)?|\d+\s+bulan)\b", line["text"])
            if match and keyword_in(line["text"], ("masa kerja", "bekerja selama")):
                return clean_label_value(match.group(1))
        return None

    def _find_institution(self, lines: list[dict[str, Any]]) -> str | None:
        for line in lines[:15]:
            text = line["text"]
            if keyword_in(text, self.config.institution_keywords):
                match = re.search(r"(?i)\b(?:PT|CV)\.?\s+.+", text)
                value = clean_label_value(match.group(0) if match else text)
                value = re.split(r"(?i)\s+(?:menyatakan|menerangkan|dengan sebenar)", value, maxsplit=1)[0]
                return clean_label_value(value)
        labeled = self._find_by_labels(lines, ("perusahaan", "company"))
        if labeled:
            return labeled
        return None

    def _find_signer(self, lines: list[dict[str, Any]]) -> tuple[str | None, str | None]:
        tail = [line["text"] for line in lines[-15:]]
        candidates = [clean_label_value(line) for line in tail if self._looks_like_person(line)]
        signer_name = candidates[-1] if candidates else None
        signer_position = None
        if signer_name:
            signer_index = next((idx for idx, line in enumerate(tail) if signer_name in line), len(tail) - 1)
            nearby = tail[max(0, signer_index - 3) : signer_index + 4]
            for line in nearby:
                if (
                    keyword_in(line, ("hrd", "direktur", "manager", "kepala", "pimpinan", "human resources"))
                    and line != signer_name
                    and len(line) < 80
                ):
                    signer_position = clean_label_value(line)
                    break
        return signer_name, signer_position

    def _find_purpose(self, lines: list[dict[str, Any]]) -> str | None:
        for index, item in enumerate(lines):
            if not keyword_in(item["text"], self.config.purpose_keywords):
                continue
            sentence = " ".join(line["text"] for line in lines[index : index + 3])
            sentence = re.split(r"(?<=[.;])\s+", sentence, maxsplit=1)[0]
            return clean_label_value(sentence)
        return None

    def _looks_like_person(self, value: str) -> bool:
        cleaned = clean_label_value(value)
        if not cleaned or len(cleaned) < 4:
            return False
        if not cleaned[0].isupper() or ":" in cleaned:
            return False
        if re.search(r"[^A-Za-zÀ-ÿ .'`-]", cleaned):
            return False
        if any(char.isdigit() for char in cleaned):
            return False
        if keyword_in(cleaned, self.config.title_keywords + self.config.institution_keywords):
            return False
        if cleaned.isupper():
            return False
        if keyword_in(cleaned, ("cafe", "gelato", "tube ice", "group", "resources development")):
            return False
        words = cleaned.split()
        return 2 <= len(words) <= 5 and sum(word[:1].isupper() for word in words) >= 1
