"""Worker configuration."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gcp_project_id: str = "labyra-app-dev"
    gcp_region: str = "asia-southeast1"
    firebase_bucket: str = ""

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5-20250929"
    anthropic_max_tokens: int = 4096

    default_locale: str = "en"

    # R160-spectra-3c-hotfix3: TGA + DSC + OCP added
    analysis_version: str = "spectra-4b-1.5.0"

    delete_raw_after_analyze: bool = False
    max_peaks: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()
