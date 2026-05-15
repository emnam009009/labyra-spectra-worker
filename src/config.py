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

    # R167-B2: Mistral OCR for paper processing pipeline
    mistral_api_key: str = ""

    # R167-B3: Voyage AI embedding + Pinecone vector store
    voyage_api_key: str = ""
    pinecone_api_key: str = ""
    pinecone_index_name: str = "labyra-papers"

    # R167-B4: Enrichment step toggle (OFF by default — cost ~$0.10/paper)
    # Set true via ENABLE_ENRICHMENT=true env to unlock ~35% retrieval boost.
    enable_enrichment: bool = False

    # R167-B5: Citation extraction polite-pool identification for Crossref + OpenAlex
    # Empty string falls back to default 'labyra-platform@github.io' in clients.
    crossref_polite_mailto: str = ""
    openalex_polite_mailto: str = ""

    default_locale: str = "en"

    # R160-spectra-3c-hotfix3: TGA + DSC + OCP added
    analysis_version: str = "spectra-4b-1.5.0"

    delete_raw_after_analyze: bool = False
    max_peaks: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()
