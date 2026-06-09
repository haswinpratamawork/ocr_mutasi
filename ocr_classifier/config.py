"""Runtime configuration loaded once from .env at startup.

Mirrors ocr_mutasi/config.py: a typed pydantic-settings view of the
repo-root .env, read once per process via an lru_cache singleton. The
Azure OpenAI keys are shared with the sibling services; the OCR_* keys are
specific to this classifier.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed view of environment variables."""

    # Azure OpenAI (shared with ocr_mutasi / ocr_match)
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_api_version: str = "2025-01-01-preview"
    azure_openai_deployment: str = "gpt-4.1-mini"

    # PaddleOCR upstream (the Google-Cloud markdown OCR service)
    ocr_endpoint_url: str
    ocr_api_key: str
    ocr_skip_orientation: bool = False
    ocr_timeout_s: float = 120.0

    # Service
    app_host: str = "0.0.0.0"
    # The actual bind port comes from run_api.sh / the uvicorn --port flag, not
    # this default (the shared root .env may carry an APP_PORT for another
    # service). ocr_classifier runs on 8000.
    app_port: int = 8000

    # Limits
    llm_request_timeout_s: float = 120.0
    max_pdf_bytes: int = 20_000_000
    max_files: int = 50
    # Cap the OCR text handed to the LLM. The distinguishing title/header
    # tokens appear early, so this bounds tokens/cost without losing signal.
    max_classify_chars: int = 8000
    # How many OCR calls to run concurrently in a batch (each call is slow,
    # ~10s), so a batch isn't a serial wait.
    batch_ocr_concurrency: int = 4

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton accessor — read .env once per process."""
    return Settings()  # type: ignore[call-arg]
