#!/usr/bin/env python3
"""LLM text-only field matching fallback for Surat Keterangan Kerja."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


BASE_DIR = Path(__file__).resolve().parent


def env_file_candidates(explicit_path: Path | None = None) -> list[Path]:
    if explicit_path:
        return [explicit_path]
    candidates = [
        BASE_DIR.parent / ".env",  # repo root — centralized config
        Path.cwd() / ".env",
    ]
    result = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(path)
    return result


@dataclass(frozen=True)
class LLMConfig:
    endpoint: str
    api_key: str
    api_version: str
    deployment: str

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> "LLMConfig":
        for candidate in env_file_candidates(env_path):
            load_env_file(candidate)
        required = {
            "endpoint": "AZURE_OPENAI_ENDPOINT",
            "api_key": "AZURE_OPENAI_API_KEY",
            "api_version": "AZURE_OPENAI_API_VERSION",
            "deployment": "AZURE_OPENAI_DEPLOYMENT",
        }
        values = {field: os.environ.get(name, "").strip() for field, name in required.items()}
        missing = [required[field] for field, value in values.items() if not value]
        if missing:
            raise RuntimeError(
                "LLM text fallback is not configured. Missing: "
                + ", ".join(missing)
                + ". Set them in the repo-root .env (see .env.example)."
            )
        return cls(**values)

    @property
    def chat_completions_url(self) -> str:
        endpoint = self.endpoint.rstrip("/")
        return f"{endpoint}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class LLMTextMatcher:
    """Use LLM to match SKK text into fixed fields."""

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
                        "You map OCR text from Indonesian Surat Keterangan Kerja documents "
                        "into fixed JSON fields. Return strict JSON only."
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
  "document_number": "string or null",
  "document_date": "YYYY-MM-DD or null",
  "worker_name": "string or null",
  "employee_id": "string or null",
  "position": "string or null",
  "department": "string or null",
  "employment_status": "string or null",
  "start_date": "YYYY-MM-DD or null",
  "end_date": "YYYY-MM-DD or null",
  "tenure": "string or null",
  "institution_name": "string or null",
  "signer_name": "string or null",
  "signer_position": "string or null",
  "purpose": "string or null",
  "confidence_notes": ["short note strings"]
}}

Rules:
- Use semantic similarity for Indonesian and English labels.
- Preserve names and document numbers as written, but trim labels and punctuation.
- Use existing parser values when they are already correct.
- Use null for fields that are not visible in the text.
""".strip()

    def _error_message(self, response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return f"{response.status_code} {response.text}"
        error = payload.get("error", {})
        return f"{response.status_code} {error.get('message') or response.text}"


def normalize_llm_document(raw: dict[str, Any], source_file: str, page_number: int) -> dict[str, Any]:
    return {
        "source_file": f"{source_file}#page-{page_number}",
        "page_number": page_number,
        "document_number": raw.get("document_number"),
        "document_date": raw.get("document_date"),
        "worker_name": raw.get("worker_name"),
        "employee_id": raw.get("employee_id"),
        "position": raw.get("position"),
        "department": raw.get("department"),
        "employment_status": raw.get("employment_status"),
        "start_date": raw.get("start_date"),
        "end_date": raw.get("end_date"),
        "tenure": raw.get("tenure"),
        "institution_name": raw.get("institution_name"),
        "signer_name": raw.get("signer_name"),
        "signer_position": raw.get("signer_position"),
        "purpose": raw.get("purpose"),
        "confidence_notes": raw.get("confidence_notes") or [],
        "extraction_method": "ocr_text_llm_match",
    }
