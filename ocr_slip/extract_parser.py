#!/usr/bin/env python3
"""Extract and summarize salary-slip PDFs.

The script keeps PDF extraction separate from payroll inference so new salary
slip layouts can be supported by changing keyword rules instead of rewriting
the pypdfium2 extraction layer.
"""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pypdfium2 as pdfium


MONEY_RE = re.compile(
    r"(?<![\w/])(?:rp|idr)?\.?\s*([-+]?(?:\d{1,3}(?:[.,]\d{3})+|\d+)(?:[.,]\d{2})?)(?![\w/])",
    re.IGNORECASE,
)
LABEL_PREFIX_RE = re.compile(r"^(?:[\s\-•*–—]+|\d{3,5}\s+|TOTAL\.[A-Z]+\s+)")


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
    """Keywords used by the inference layer.

    Extend these lists for a new company or language before changing parser
    logic. Matching is case-insensitive.
    """

    earnings_sections: tuple[str, ...] = ("upah", "penghasilan", "pendapatan", "gaji", "earnings")
    company_contribution_sections: tuple[str, ...] = (
        "kontribusi perusahaan",
        "employer contribution",
        "non cash benefit",
        "benefit tidak tunai",
    )
    deduction_sections: tuple[str, ...] = ("potongan", "deduction", "deductions")
    total_keywords: tuple[str, ...] = ("total", "jumlah")
    net_pay_keywords: tuple[str, ...] = (
        "bank transfer",
        "take home pay",
        "thp",
        "net pay",
        "net salary",
        "dibayarkan",
        "diterima",
        "bersih",
        "neto",
        "bruto",
        "total penghasilan bruto",
        "penghasilan bruto",
        "take-home",
        "take home",
    )
    incentive_keywords: tuple[str, ...] = (
        "tunjangan",
        "allowance",
        "incentive",
        "insentif",
        "bonus",
        "kompensasi",
        "lembur",
        "overtime",
        "subsidi",
        "subsidy",
        "premium",
        "uang makan",
        "penerimaan lainnya",
    )
    tax_keywords: tuple[str, ...] = ("pph", "pajak", "tax")
    earning_item_keywords: tuple[str, ...] = (
        "gaji",
        "upah",
        "salary",
        "wage",
        "earnings",
        "pendapatan",
        "penghasilan",
        "penerimaan",
        "kredit",
        "inflow",
        "compensation",
        "tunjangan",
        "allowance",
        "incentive",
        "insentif",
        "bonus",
        "kompensasi",
        "lembur",
        "overtime",
        "subsidi",
        "subsidy",
        "premium",
        "uang makan",
    )
    deduction_item_keywords: tuple[str, ...] = (
        "potongan",
        "deduction",
        "outflow",
        "debit",
        "pengurangan",
        "pemotongan",
        "pajak",
        "pph",
        "tax",
        "bpjs",
        "iuran",
        "jht",
        "jaminan",
        "pensiun",
        "withholding",
        "koperasi",
        "opt-in",
        "swadana",
    )
    worker_name_keywords: tuple[str, ...] = ("nama", "name", "employee", "karyawan", "pekerja")
    institution_keywords: tuple[str, ...] = (
        "pt.",
        "pt ",
        "cv.",
        "cv ",
        "persero",
        "bank",
        "company",
        "perusahaan",
        "institusi",
    )

    @classmethod
    def from_json(cls, path: Path) -> "ParserConfig":
        """Load keyword overrides from JSON.

        The JSON may contain any ParserConfig field as a list of strings. Values
        replace the defaults for that field.
        """

        raw = json.loads(path.read_text(encoding="utf-8"))
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"Unknown config field(s): {', '.join(unknown)}")
        return cls(**{key: tuple(value) for key, value in raw.items()})


@dataclass
class LineItem:
    label: str
    amount: int | float
    raw_amount: str
    section: str | None
    page: int
    line_number: int
    raw_line: str


@dataclass
class SalarySummary:
    source_file: str
    worker_name: str | None
    institution: str | None
    paid_salary_total: int | float | None
    gross_income_total: int | float | None
    deduction_total: int | float | None
    incentive_total: int | float
    tax_cutoff_total: int | float
    other_cutoff_total: int | float
    earnings: list[LineItem] = field(default_factory=list)
    incentives: list[LineItem] = field(default_factory=list)
    deductions: list[LineItem] = field(default_factory=list)
    tax_cutoffs: list[LineItem] = field(default_factory=list)
    other_cutoffs: list[LineItem] = field(default_factory=list)
    company_contributions: list[LineItem] = field(default_factory=list)
    confidence_notes: list[str] = field(default_factory=list)


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def keyword_in(text: str, keywords: Iterable[str]) -> bool:
    lowered = text.casefold()
    return any(keyword.casefold() in lowered for keyword in keywords)


def parse_money(raw: str) -> int | float:
    """Parse common Indonesian and international money formats."""

    value = raw.strip()
    value = value.strip("()")
    value = re.sub(r"(?i)^(rp|idr)\.?\s*", "", value)

    if "," in value and "." in value:
        decimal_sep = "," if value.rfind(",") > value.rfind(".") else "."
        thousand_sep = "." if decimal_sep == "," else ","
        value = value.replace(thousand_sep, "")
        value = value.replace(decimal_sep, ".")
        number = float(value)
    elif re.fullmatch(r"[-+]?\d{1,3}(?:[.,]\d{3})+", value):
        number = int(value.replace(".", "").replace(",", ""))
    elif re.fullmatch(r"[-+]?\d+[.,]\d{2}", value):
        number = float(value.replace(",", "."))
    else:
        number = int(value)

    return int(number) if isinstance(number, float) and number.is_integer() else number


def money_matches(line: str) -> list[re.Match[str]]:
    matches: list[re.Match[str]] = []
    for match in MONEY_RE.finditer(line):
        before = line[match.start() - 1] if match.start() > 0 else ""
        after = line[match.end()] if match.end() < len(line) else ""
        before_before = line[match.start() - 2] if match.start() > 1 else ""
        after_after = line[match.end() + 1] if match.end() + 1 < len(line) else ""

        part_of_date = (
            before in "./-" and before_before.isdigit()
        ) or (
            after in "./-" and after_after.isdigit()
        )
        part_of_percent = after == "%"
        raw_value = match.group(1)
        has_currency = re.search(r"(?i)(rp|idr)", match.group(0)) is not None
        has_thousands = bool(re.search(r"\d[.,]\d{3}", raw_value))
        if not part_of_date and not part_of_percent and (has_currency or has_thousands):
            matches.append(match)
    return matches


def clean_label(label: str) -> str:
    label = normalize_space(label)
    label = re.sub(r"(?i)\b(rp|idr)\.?\s*$", "", label).strip()
    label = re.sub(r"\s*[:=]\s*$", "", label).strip()
    label = LABEL_PREFIX_RE.sub("", label).strip()
    return normalize_space(label.strip(" :|-()"))


def amount_at_line_end(line: str) -> tuple[str, int | float] | None:
    matches = money_matches(line)
    if not matches:
        return None
    match = matches[-1]
    if normalize_space(line[match.end() :]):
        return None
    return match.group(1), parse_money(match.group(1))


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


def text_quality_needs_ocr(extracted: dict[str, Any], config: ParserConfig | None = None) -> bool:
    config = config or ParserConfig()
    text = "\n".join(page["text"] for page in extracted["pages"])
    lines = [line for page in extracted["pages"] for line in page["lines"]]
    page_count = max(extracted["page_count"], 1)
    total_chars = len(text)
    avg_chars = total_chars / page_count
    money_count = sum(len(money_matches(line)) for line in lines)
    has_keyword = any(
        keyword_in(
            line,
            config.earning_item_keywords
            + config.deduction_item_keywords
            + config.net_pay_keywords
            + config.worker_name_keywords,
        )
        for line in lines
    )

    return (
        total_chars < 150
        or avg_chars < 80
        or len(lines) < 5
        or money_count == 0
        or not has_keyword
    )


class SalarySlipAnalyzer:
    """Infer payroll fields from extracted text."""

    def __init__(self, config: ParserConfig | None = None) -> None:
        self.config = config or ParserConfig()

    def analyze(self, extracted: dict[str, Any]) -> SalarySummary:
        lines = self._flatten_lines(extracted)
        items = self._extract_line_items(lines)

        earnings = [item for item in items if item.section == "earnings" and not self._is_total(item)]
        deductions = [item for item in items if item.section == "deductions" and not self._is_total(item)]
        contributions = [
            item for item in items if item.section == "company_contributions" and not self._is_total(item)
        ]
        incentives = [item for item in earnings if keyword_in(item.label, self.config.incentive_keywords)]
        tax_cutoffs = [item for item in deductions if keyword_in(item.label, self.config.tax_keywords)]
        other_cutoffs = [item for item in deductions if item not in tax_cutoffs]

        gross_income_total = self._find_total(items, "earnings")
        deduction_total = self._positive_money(self._find_total(items, "deductions"))
        paid_salary_total = self._find_net_pay(lines, gross_income_total, deduction_total)

        notes: list[str] = []
        if paid_salary_total is None and gross_income_total is not None and deduction_total is not None:
            paid_salary_total = gross_income_total - deduction_total
            notes.append("paid_salary_total inferred as gross_income_total - deduction_total.")
        if extracted.get("warnings"):
            notes.extend(extracted["warnings"])
        if not deductions and deduction_total:
            notes.append("Deduction total found, but individual deduction rows were not classified.")

        return SalarySummary(
            source_file=extracted["source_file"],
            worker_name=self._find_worker_name(lines),
            institution=self._find_institution(lines),
            paid_salary_total=paid_salary_total,
            gross_income_total=gross_income_total,
            deduction_total=deduction_total,
            incentive_total=sum(item.amount for item in incentives),
            tax_cutoff_total=sum(abs(item.amount) for item in tax_cutoffs),
            other_cutoff_total=sum(abs(item.amount) for item in other_cutoffs),
            earnings=earnings,
            incentives=incentives,
            deductions=deductions,
            tax_cutoffs=tax_cutoffs,
            other_cutoffs=other_cutoffs,
            company_contributions=contributions,
            confidence_notes=notes,
        )

    def _flatten_lines(self, extracted: dict[str, Any]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for page in extracted["pages"]:
            for line_index, line in enumerate(page["lines"], start=1):
                result.append(
                    {
                        "page": page["page_number"],
                        "line_number": line_index,
                        "text": line,
                    }
                )
        return result

    def _extract_line_items(self, lines: list[dict[str, Any]]) -> list[LineItem]:
        section: str | None = None
        pending_labels: deque[tuple[str, str | None, dict[str, Any]]] = deque()
        items: list[LineItem] = []

        for line_info in lines:
            line = line_info["text"]
            matches = money_matches(line)
            detected_section = self._detect_section(line)
            if detected_section and (not matches or detected_section in {"mixed", "company_contributions"}):
                section = detected_section

            if not matches:
                queued_label = self._label_from_amountless_line(line)
                if queued_label:
                    pending_labels.append(
                        (queued_label, self._classify_label(queued_label, section), line_info)
                    )
                continue

            if "=" in line and len(matches) > 1 and self._looks_like_formula(line):
                continue

            item_specs = self._line_item_specs(line, matches, section, pending_labels)
            for label, raw_amount, parsed_amount, item_section in item_specs:
                if (
                    not label
                    or self._looks_like_date(label)
                    or self._is_non_payroll_label(label)
                    or item_section == "net_pay"
                ):
                    continue

                items.append(
                    LineItem(
                        label=label,
                        amount=parsed_amount,
                        raw_amount=raw_amount,
                        section=item_section,
                        page=line_info["page"],
                        line_number=line_info["line_number"],
                        raw_line=line,
                    )
                )

        return items

    def _detect_section(self, line: str) -> str | None:
        lowered = line.casefold()
        if keyword_in(lowered, self.config.company_contribution_sections):
            return "company_contributions"

        earning_signal = keyword_in(lowered, self.config.earnings_sections) or any(
            word in lowered for word in ("kredit", "inflow", "compensation", "penerimaan")
        )
        deduction_signal = keyword_in(lowered, self.config.deduction_sections) or any(
            word in lowered for word in ("debit", "outflow", "pengurangan", "pemotongan")
        )

        if earning_signal and deduction_signal:
            return "mixed"
        if deduction_signal:
            return "deductions"
        if earning_signal:
            return "earnings"
        return None

    def _line_item_specs(
        self,
        line: str,
        matches: list[re.Match[str]],
        section: str | None,
        pending_labels: deque[tuple[str, str | None, dict[str, Any]]],
    ) -> list[tuple[str, str, int | float, str | None]]:
        specs: list[tuple[str, str, int | float, str | None]] = []

        if len(matches) == 1:
            match = matches[0]
            label = clean_label(line[: match.start()])
            if not label and pending_labels:
                label, queued_section, _ = pending_labels.popleft()
                item_section = queued_section or self._classify_label(label, section)
            else:
                item_section = self._classify_label(label, section)

            specs.append((label, match.group(1), parse_money(match.group(1)), item_section))
            return specs

        for index, match in enumerate(matches):
            start = 0 if index == 0 else matches[index - 1].end()
            label = clean_label(line[start : match.start()])
            item_section = self._classify_label(label, section, mixed_index=index)
            specs.append((label, match.group(1), parse_money(match.group(1)), item_section))

        return specs

    def _label_from_amountless_line(self, line: str) -> str | None:
        label = clean_label(line)
        if not label:
            return None
        lowered = label.casefold()
        if lowered in {
            "pendapatan",
            "penghasilan",
            "penerimaan",
            "earnings",
            "potongan",
            "pengurangan",
            "deductions",
            "outflows",
            "inflows",
            "kredit",
            "debit",
        }:
            return None
        if any(
            keyword in lowered
            for keyword in (
                "komponen",
                "jumlah",
                "slip",
                "laporan",
                "statement",
                "rekap",
                "audit",
                "employee name",
                "nama ",
                "tax status",
                "pay cycle",
                "status",
                "jabatan",
                "grade",
                "cost center",
            )
        ):
            return None
        if keyword_in(label, self.config.earning_item_keywords + self.config.deduction_item_keywords):
            return label
        if self._looks_like_person(label):
            return None
        if any(
            keyword in lowered
            for keyword in (
                "terbilang",
                "periode",
                "alamat",
                "manager",
                "sistem",
                "dokumen",
                "informasi",
                "payment method",
                "rekening",
            )
        ):
            return None
        return None

    def _classify_label(
        self,
        label: str,
        current_section: str | None,
        mixed_index: int | None = None,
    ) -> str | None:
        lowered = label.casefold()
        if keyword_in(lowered, self.config.net_pay_keywords):
            return "net_pay"
        if keyword_in(lowered, self.config.company_contribution_sections):
            return "company_contributions"
        if current_section == "company_contributions":
            return "company_contributions"

        earning_total = any(
            word in lowered
            for word in ("gross", "bruto", "pendapatan", "penghasilan", "earnings", "kredit", "inflow")
        )
        deduction_total = any(
            word in lowered
            for word in ("potongan", "deduction", "debit", "pengurangan", "outflow")
        )

        earning_label = keyword_in(lowered, self.config.earning_item_keywords)
        deduction_label = keyword_in(lowered, self.config.deduction_item_keywords)
        earning_first = label.casefold().startswith(
            ("tunjangan", "allowance", "subsidi", "subsidy", "bonus", "insentif", "incentive")
        )

        if keyword_in(lowered, self.config.total_keywords):
            if deduction_total and not earning_total:
                return "deductions"
            if earning_total and not deduction_total:
                return "earnings"

        if current_section == "mixed" and mixed_index is not None:
            if deduction_label and not earning_first:
                return "deductions"
            if earning_label:
                return "earnings"
            if earning_total:
                return "earnings"
            if deduction_total:
                return "deductions"
            return "earnings" if mixed_index == 0 else "deductions"

        if deduction_label and not earning_first:
            return "deductions"
        if earning_label and not label.casefold().startswith(("pph", "pajak", "tax")):
            return "earnings"
        if current_section == "mixed":
            return None
        return current_section

    def _looks_like_formula(self, line: str) -> bool:
        return "=" in line and any(operator in line for operator in ("-", "+"))

    def _is_non_payroll_label(self, label: str) -> bool:
        lowered = label.casefold()
        return any(
            keyword in lowered
            for keyword in (
                "nomor slip",
                "no. referensi",
                "referensi",
                "periode",
                "date of joining",
                "pay date",
                "disbursement date",
                "rekening",
                "bank account",
                "status pajak",
                "ptkp",
                "terbilang",
                "jakarta,",
                "surabaya,",
                "periode buku",
            )
        )

    def _find_total(self, items: list[LineItem], section: str) -> int | float | None:
        totals = []
        for item in items:
            if item.section != section or not keyword_in(item.label, self.config.total_keywords):
                continue
            label = item.label.casefold()
            if section == "earnings" and any(
                keyword in label
                for keyword in ("pendapatan", "penghasilan", "earnings", "bruto", "gross", "kredit", "inflow", "upah")
            ):
                totals.append(item)
            elif section == "deductions" and any(
                keyword in label
                for keyword in ("potongan", "deduction", "debit", "pengurangan", "outflow")
            ):
                totals.append(item)
            elif item.label.casefold() in {"total", "total:"}:
                totals.append(item)
        return totals[-1].amount if totals else None

    def _positive_money(self, value: int | float | None) -> int | float | None:
        return abs(value) if isinstance(value, (int, float)) else value

    def _find_net_pay(
        self,
        lines: list[dict[str, Any]],
        gross_income_total: int | float | None,
        deduction_total: int | float | None,
    ) -> int | float | None:
        for index, line_info in enumerate(lines):
            line = line_info["text"]
            if keyword_in(line, self.config.net_pay_keywords):
                window_lines = [line]
                window_lines.extend(item["text"] for item in lines[index + 1 : index + 4])
                window_lines.extend(item["text"] for item in lines[max(0, index - 3) : index])
                values = [
                    parse_money(match.group(1))
                    for window_line in window_lines
                    for match in money_matches(window_line)
                ]
                if values:
                    if gross_income_total in values and deduction_total in values and len(values) >= 3:
                        return next(
                            value
                            for value in values
                            if value not in {gross_income_total, deduction_total}
                        )
                    non_total_values = [
                        value for value in values if value not in {gross_income_total, deduction_total}
                    ]
                    if non_total_values:
                        return non_total_values[0]
                    return values[0]

        return None

    def _find_worker_name(self, lines: list[dict[str, Any]]) -> str | None:
        stop_words = (
            " NIK",
            " No.",
            " No ",
            " Tax Status",
            " Pay Cycle",
            " Shift",
            " Jabatan",
            " NPP",
            " Grade",
            " Cost Center",
            " Periode",
            " Status",
            " ID Pekerja",
        )
        name_patterns = (
            r"(?i)\bNama(?:\s+(?:Lengkap|Karyawan|Pegawai))?\s*:?\s*(.+)",
            r"(?i)\bEmployee\s*:\s*(?:\d+\s*[-–—]\s*)?(.+)",
            r"(?i)\bEmployee\s+Name\s*:?\s*(.+)",
            r"(?i)\bName\s*:?\s*(.+)",
        )

        for line_info in lines:
            line = line_info["text"]
            for pattern in name_patterns:
                match = re.search(pattern, line)
                if not match:
                    continue
                value = normalize_space(match.group(1))
                for stop_word in stop_words:
                    if stop_word in value:
                        value = normalize_space(value.split(stop_word, 1)[0])
                value = value.strip(" :|-")
                if self._looks_like_person(value) or self._looks_like_explicit_name(value):
                    return value

        for line_info in lines[:8]:
            line = line_info["text"]
            if "/" in line:
                candidate = normalize_space(line.split("/", 1)[0])
                if self._looks_like_person(candidate):
                    return candidate

        ignored_people = {"manager", "penerima"}
        candidates = []
        for line_info in lines:
            line = line_info["text"]
            if line.casefold() not in ignored_people and self._looks_like_person(line):
                candidates.append(line)
        if candidates:
            return candidates[-1]
        return None

    def _find_institution(self, lines: list[dict[str, Any]]) -> str | None:
        company_candidates: list[str] = []
        for line_info in lines:
            line = line_info["text"]
            if keyword_in(line, self.config.institution_keywords):
                company_candidates.append(line)
        if company_candidates:
            pt_candidates = [line for line in company_candidates if re.search(r"(?i)\b(pt|cv)\.?\b", line)]
            selected = pt_candidates[0] if pt_candidates else company_candidates[0]
            return self._clean_institution(selected)

        for line_info in reversed(lines[-8:]):
            line = line_info["text"]
            if (
                1 <= len(line.split()) <= 3
                and not money_matches(line)
                and not keyword_in(line, self.config.net_pay_keywords)
            ):
                return line
        return lines[0]["text"] if lines else None

    def _clean_institution(self, text: str) -> str:
        value = normalize_space(text)
        for marker in (" Periode", " Karyawan", " Jabatan", " Status"):
            if marker in value:
                value = normalize_space(value.split(marker, 1)[0])
        value = re.sub(r"(?i)^(company\s+name|nama\s+perusahaan)\s*:\s*", "", value).strip()
        return value.strip(" :-")

    def _is_total(self, item: LineItem) -> bool:
        return keyword_in(item.label, self.config.total_keywords)

    def _looks_like_person(self, text: str) -> bool:
        if any(char.isdigit() for char in text) or any(char in text for char in ":/|"):
            return False
        if keyword_in(text, self.config.earning_item_keywords + self.config.deduction_item_keywords):
            return False
        if keyword_in(
            text,
            (
                "staff",
                "manager",
                "supervisor",
                "karyawan",
                "tetap",
                "akuntansi",
                "department",
                "designation",
            ),
        ):
            return False
        words = text.split()
        if not 2 <= len(words) <= 5:
            return False
        return not keyword_in(text, self.config.institution_keywords + self.config.earnings_sections)

    def _looks_like_explicit_name(self, text: str) -> bool:
        if any(char.isdigit() for char in text) or any(char in text for char in ":/|"):
            return False
        words = text.split()
        if not 1 <= len(words) <= 6:
            return False
        return not keyword_in(
            text,
            self.config.institution_keywords
            + self.config.earnings_sections
            + self.config.deduction_sections,
        )

    def _looks_like_date(self, text: str) -> bool:
        return bool(re.fullmatch(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", text.strip()))


def summary_to_jsonable(summary: SalarySummary) -> dict[str, Any]:
    return asdict(summary)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def iter_pdfs(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        return [input_path]
    return sorted(path for path in input_path.rglob("*") if path.suffix.lower() == ".pdf")
