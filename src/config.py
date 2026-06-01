"""Worker configuration."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", ".env.local"), extra="ignore")

    gcp_project_id: str = "labyra-app-dev"
    gcp_region: str = "asia-southeast1"
    firebase_bucket: str = ""

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5-20250929"
    anthropic_max_tokens: int = 4096
    # R177-1a-gemini-infra: Gemini for metadata extract + chunk enrichment
    # (cheaper alternative to Haiku 4.5 for structured JSON tasks).
    # Sonnet analyzer in src/ai/analyzer.py intentionally stays on Anthropic
    # for scientific output reliability + audit trail (provider isolation).
    gemini_api_key: str = ""
    gemini_model_metadata: str = "gemini-3-flash-preview"  # @r179-4-applied
    gemini_model_enrich: str = "gemini-3-flash-preview"
    gemini_max_tokens_metadata: int = 1536  # R224: room for abstract in JSON
    gemini_max_tokens_enrich: int = 200
    # R178-3: domain classification (taxonomy v1, ~$0.001/paper)
    # @r178-3-applied
    gemini_model_classify: str = "gemini-3-flash-preview"
    gemini_max_tokens_classify: int = 300

    # R223: Pre-translate Lop 1 (abstract/conclusion/headings) — cheap Flash
    gemini_model_pretranslate: str = "gemini-3-flash-preview"
    gemini_max_tokens_pretranslate: int = 2048
    # R177-1a Google Books API for book/textbook metadata resolution
    # (Crossref doesn't index books → fallback path for documentType='book')
    books_api_key: str = ""
    # R184: Materials Project API for crystal structure + electronic props sync
    mp_api_key: str = ""

    # R167-B2: Mistral OCR for paper processing pipeline
    mistral_api_key: str = ""

    # R221: OCR engine selection + Datalab hosted Marker (paid cloud, async poll).
    # ocr_engine "mistral" (default) | "datalab"; ocr_fallback e.g. "mistral".
    ocr_engine: str = "mistral"
    ocr_fallback: str = ""
    datalab_api_key: str = ""
    datalab_marker_url: str = "https://www.datalab.to/api/v1/marker"
    datalab_use_llm: bool = False
    datalab_langs: str = ""

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
    openalex_api_key: str = ""  # OpenAlex requires a key since 2026-02-13 (free)

    default_locale: str = "en"

    # R160-spectra-3c-hotfix3: TGA + DSC + OCP added
    analysis_version: str = "spectra-4b-1.5.0"

    delete_raw_after_analyze: bool = False
    max_peaks: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()
