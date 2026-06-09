"""Runtime configuration loaded once from .env at startup."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed view of environment variables."""

    # Azure OpenAI
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_api_version: str = "2025-01-01-preview"
    azure_openai_deployment: str = "gpt-4.1-mini"

    # Upstream services
    ocr_slip_url: str = "http://127.0.0.1:8200"
    ocr_mutasi_url: str = "http://127.0.0.1:8300"

    # Service
    app_host: str = "0.0.0.0"
    app_port: int = 8400

    # Timeouts & limits
    llm_request_timeout_s: float = 60.0  # reserved for a future LLM-tiebreak path
    upstream_timeout_s: float = 120.0

    # Matching tolerance in absolute rupiah. The user's domain guarantees
    # slip and credit amounts agree to the rupiah for genuine pairs, so the
    # default is essentially "exact match with float-safety wiggle". Bump
    # to e.g. 1000 if you have cents-rounding or fee adjustments.
    match_amount_tolerance_rp: float = 1.0

    max_files: int = 50

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
