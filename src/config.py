"""Worker configuration via Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven config. Cloud Run injects via env vars."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # GCP
    gcp_project_id: str = "labyra-app-dev"
    gcp_region: str = "asia-southeast1"
    firebase_bucket: str = ""  # e.g. "labyra-app-dev.firebasestorage.app"

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5-20250929"  # Sonnet 4.6 alias
    anthropic_max_tokens: int = 4096

    # Locale fallback
    default_locale: str = "en"

    # Analysis version (bump when prompt/parser changes meaningfully)
    analysis_version: str = "xrd-1.0.0"

    # Behavior
    delete_raw_after_analyze: bool = False  # GCS lifecycle handles this; worker just flags
    max_peaks: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()
