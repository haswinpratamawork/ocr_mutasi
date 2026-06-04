#!/usr/bin/env python3
"""LLM text-only field matching fallback."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


BASE_DIR = Path(__file__).resolve().parent
MONEY_FIELD_NAMES = {"total_paid", "pokok", "incentive", "deduction"}
AMOUNT_RE = re.compile(r"([-+]?(?:\d{1,3}(?:[.,:]\d{3})+|\d{7,})(?:[.,]\d{2})?)")


@dataclass(frozen=True)
class LLMConfig:
    endpoint: str
    api_key: str
    api_version: str
    deployment: str

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> "LLMConfig":
        load_env_file(env_path or BASE_DIR / ".env")
        required = {
            "endpoint": "LLM_ENDPOINT",
            "api_key": "LLM_API_KEY",
            "api_version": "LLM_API_VERSION",
            "deployment": "LLM_DEPLOYMENT",
        }
        values = {field: os.environ.get(name, "").strip() for field, name in required.items()}
        missing = [required[field] for field, value in values.items() if not value]
        if missing:
            raise RuntimeError(
                "LLM text fallback is not configured. Missing: "
                + ", ".join(missing)
                + ". Create a local .env file from .env.example."
            )
        return cls(**values)

    @property
    def chat_completions_url(self) -> str:
        endpoint = self.endpoint.rstrip("/")
        return (
            f"{endpoint}/openai/deployments/{self.deployment}/chat/completions"
            f"?api-version={self.api_version}"
        )


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def normalize_money_value(value: Any) -> int | float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return int(value) if isinstance(value, float) and value.is_integer() else value
    text = str(value)
    text = re.sub(r"(?i)\b(rp|idr)\b", "", text)
    text = text.replace(" ", "").replace(":", ".").strip()
    negative = text.startswith("-") or (text.startswith("(") and text.endswith(")"))
    text = text.strip("-()")
    if not text:
        return None
    if "," in text and "." in text:
        final_group = re.split(r"[,.]", text)[-1]
        if len(final_group) == 3:
            text = text.replace(".", "").replace(",", "")
        else:
            decimal_sep = "," if text.rfind(",") > text.rfind(".") else "."
            thousand_sep = "." if decimal_sep == "," else ","
            text = text.replace(thousand_sep, "").replace(decimal_sep, ".")
    elif ":" in text:
        text = text.replace(":", "")
    elif re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", text):
        text = text.replace(".", "").replace(",", "")
    elif re.fullmatch(r"\d+[.,]\d{2}", text):
        text = text.replace(",", ".")
    try:
        number = float(text)
    except ValueError:
        return None
    if negative:
        number *= -1
    return int(number) if number.is_integer() else number


class LLMTextMatcher:
    """Use LLM to match similar text labels into fixed fields."""

    def __init__(self, config: LLMConfig | None = None, timeout: int = 60) -> None:
        self.config = config or LLMConfig.from_env()
        self.timeout = timeout

    def match_page(
        self,
        page_text: str,
        source_file: str,
        page_number: int,
        existing: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You map OCR text from Indonesian payroll documents into fixed JSON fields. "
                        "Use semantic similarity for labels. Return strict JSON only."
                    ),
                },
                {"role": "user", "content": self._prompt(page_text, page_number, existing or {})},
            ],
            "temperature": 0,
            "max_tokens": 1200,
            "response_format": {"type": "json_object"},
        }
        response = requests.post(
            self.config.chat_completions_url,
            headers={"api-key": self.config.api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"LLM text fallback failed: {self._error_message(response)}")
        raw = json.loads(response.json()["choices"][0]["message"]["content"])
        raw["_page_text"] = page_text
        return normalize_llm_document(raw, source_file, page_number)

    def _prompt(self, page_text: str, page_number: int, existing: dict[str, Any]) -> str:
        return f"""
Page: {page_number}

Existing parser result:
{json.dumps(existing, ensure_ascii=False)}

OCR/text content:
{page_text[:12000]}

Return this JSON object:
{{
  "extracted_date": "YYYY-MM or YYYY-MM-DD or null",
  "worker_name": "string or null",
  "institution_name": "string or null",
  "total_paid": number or null,
  "pokok": number,
  "incentive": number,
  "deduction": number,
  "confidence_notes": ["short note strings"]
}}

Rules:
- total_paid = final amount received. Similar labels: TOTAL DITERIMA, diterima karyawan, bank transfer, take home pay, net pay. If final amount is unreadable or missing, use Total Penghasilan Bruto / Bruto minus deduction.
- pokok = base/main compensation only. Similar labels: Gaji, Gaji Pokok, Upah Pokok. If multiple Gaji rows exist and there is no separate allowance/bonus amount, sum the Gaji rows as pokok.
- incentive = positive additions beside pokok, such as THR, bonus, tunjangan, allowance, overtime, meal/transport additions.
- deduction = all decreases combined into one number, including PPh/pajak/tax and any other deduction.
- Use existing parser values when they are already correct.
- If the text says a deduction row is blank or "-", count it as 0.
- OCR may confuse thousand separators, for example 49.889.125 means 49889125 and 17.503,500 means 17503500.
""".strip()

    def _error_message(self, response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return f"{response.status_code} {response.text}"
        error = payload.get("error", {})
        return f"{response.status_code} {error.get('message') or response.text}"


def normalize_llm_document(raw: dict[str, Any], source_file: str, page_number: int) -> dict[str, Any]:
    doc = {
        "source_file": f"{source_file}#page-{page_number}",
        "page_number": page_number,
        "extracted_date": raw.get("extracted_date"),
        "worker_name": raw.get("worker_name"),
        "institution_name": raw.get("institution_name"),
        "total_paid": raw.get("total_paid"),
        "pokok": raw.get("pokok"),
        "incentive": raw.get("incentive"),
        "deduction": raw.get("deduction"),
        "confidence_notes": raw.get("confidence_notes") or [],
        "extraction_method": "ocr_text_llm_match",
    }
    for field in MONEY_FIELD_NAMES:
        doc[field] = normalize_money_value(doc[field]) or 0
    postprocess_from_text(doc, raw.get("_page_text") or "")
    if not doc["total_paid"] and (doc["pokok"] or doc["incentive"] or doc["deduction"]):
        doc["total_paid"] = doc["pokok"] + doc["incentive"] - doc["deduction"]
        doc["confidence_notes"].append("total_paid inferred as pokok + incentive - deduction.")
    elif not doc["total_paid"]:
        doc["total_paid"] = None
    if doc["extracted_date"]:
        doc["source_file"] = f"{source_file}#{doc['extracted_date']}"
    return doc


def amount_from_line(line: str) -> int | float | None:
    matches = AMOUNT_RE.findall(line.replace(" ", ""))
    if not matches:
        return None
    return normalize_money_value(matches[-1])


def amounts_from_lines(page_text: str, keywords: tuple[str, ...]) -> list[int | float]:
    values = []
    for line in page_text.splitlines():
        lowered = line.casefold()
        if any(keyword in lowered for keyword in keywords):
            amount = amount_from_line(line)
            if amount is not None:
                values.append(amount)
    return values


def postprocess_from_text(doc: dict[str, Any], page_text: str) -> None:
    if not page_text:
        return

    bruto_values = amounts_from_lines(page_text, ("bruto", "total penghasilan"))
    bruto = bruto_values[-1] if bruto_values else None

    diterima_values = amounts_from_lines(page_text, ("total diterima", "diterima karyawan"))
    diterima = diterima_values[-1] if diterima_values else None

    pengurangan_values = amounts_from_lines(page_text, ("total pengurangan", "total potongan"))
    if pengurangan_values:
        doc["deduction"] = pengurangan_values[-1]
    elif "total pengurangan" in page_text.casefold():
        doc["deduction"] = 0

    gaji_values = amounts_from_lines(page_text, ("gaji", "gafi"))
    if gaji_values:
        doc["pokok"] = sum(gaji_values)

    incentive_values = amounts_from_lines(page_text, ("tunjangan", "bonus", "thr", "lembur", "penerimaan lainnya"))
    if incentive_values:
        doc["incentive"] = sum(incentive_values)
    elif bruto and gaji_values:
        doc["pokok"] = bruto if abs(sum(gaji_values) - bruto) <= 10000 else sum(gaji_values)
        doc["incentive"] = 0

    institution = institution_from_text(page_text)
    current_institution = str(doc.get("institution_name") or "")
    bad_institution = (
        not current_institution
        or len(current_institution) <= 3
        or "ptkp" in current_institution.casefold()
        or "status" in current_institution.casefold()
    )
    if institution and bad_institution:
        doc["institution_name"] = institution

    if bruto and (doc.get("deduction") or 0) == 0:
        earnings_total = (doc.get("pokok") or 0) + (doc.get("incentive") or 0)
        if earnings_total and abs(earnings_total - bruto) <= 10000:
            if doc.get("incentive"):
                doc["incentive"] = max(0, bruto - (doc.get("pokok") or 0))
            else:
                doc["pokok"] = bruto

    if bruto and (doc.get("deduction") or 0) == 0:
        if doc.get("total_paid") != bruto:
            doc["confidence_notes"].append("total_paid set from Bruto because total deductions are blank or zero.")
        doc["total_paid"] = bruto
    elif bruto and (not doc.get("total_paid") or doc["total_paid"] > bruto * 3):
        doc["total_paid"] = bruto - (doc.get("deduction") or 0)
        doc["confidence_notes"].append("total_paid corrected from Bruto because received amount was missing or OCR-malformed.")
    elif diterima and (not bruto or diterima <= bruto * 3):
        doc["total_paid"] = diterima


def institution_from_text(page_text: str) -> str | None:
    ignored = ("foto", "slip", "periode", "penerimaan", "pengurangan", "total", "brispot")
    for line in page_text.splitlines():
        clean = line.strip(" :-=;~")
        if "periode" in clean.casefold():
            clean = re.split(r"(?i)\bperiode\b", clean, maxsplit=1)[0].strip(" :-=;~")
        lowered = clean.casefold()
        if not clean or any(word in lowered for word in ignored):
            continue
        if re.search(r"\d", clean):
            continue
        if 2 <= len(clean.split()) <= 5:
            return clean
    return None
