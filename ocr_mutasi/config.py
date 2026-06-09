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

    # Service
    app_host: str = "0.0.0.0"
    app_port: int = 8300

    # Limits
    # Default 120s — the batch endpoint doubles this for cross-month calls,
    # giving 240s. Real-world batches of ~300 credits (a busy account, 12
    # months) complete in ~140s; the prior 30s default timed out and left
    # every credit unclassified.
    llm_request_timeout_s: float = 120.0
    max_pdf_bytes: int = 20_000_000

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
