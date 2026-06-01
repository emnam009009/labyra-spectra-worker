"""Extract paper metadata (title, authors, year, DOI) from OCR\'d first page.

R177-1b: migrated from Anthropic Haiku 4.5 → Gemini 3 Flash.
  Reasoning: structured JSON output, no caching benefit (single-shot per
  paper), Gemini cheaper at $0.50/$3 vs Haiku $1/$5.
  Cost: ~$0.0026/paper vs Haiku ~$0.005/paper (~50% reduction).

Pydantic schema (ExtractedMetadata) drives Gemini structured output —
SDK constrains generation to match schema, eliminating most JSON parse
errors vs the old prompt-based approach.

Returns defaults (title=\'Untitled\', authors=[], year=0, doi=\'\') on:
  - Empty/too-short input (<50 chars)
  - LLM call failure
  - JSON parse failure (defensive — SDK should prevent)
  - Schema validation failure

This is best-effort metadata — citation step + downstream search still
work with defaults.

@phase R167-B4 → R176-1a (year fallback) → R177-1b (Gemini migration)
"""
from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.config import get_settings
from src.papers._gemini_client import extract_json

logger = logging.getLogger(__name__)

METADATA_INPUT_CHAR_LIMIT = 4000  # match TS slice(0, 4000)
MIN_INPUT_CHARS = 50

# R168-3.4: year sanity bounds + fallback regex
YEAR_MIN = 1800
YEAR_MAX = 2100
_YEAR_FALLBACK_RE = re.compile(r"\b(19\d{2}|20[0-3]\d)\b")


def _coerce_year(value: Any, fallback_text: str) -> int:
    """Defensive year coercion (R168-3.4).

    Gemini structured output usually returns proper int, but defensive
    coercion stays for backward-compat + regex fallback when extract miss.
    """
    # Path 1: already valid int
    if isinstance(value, int) and not isinstance(value, bool):
        if YEAR_MIN <= value <= YEAR_MAX:
            return value
    # Path 2: string "2024" → int
    if isinstance(value, str):
        s = value.strip()
        if s.isdigit() and len(s) == 4:
            y = int(s)
            if YEAR_MIN <= y <= YEAR_MAX:
                return y
    # Path 3: regex fallback from OCR text (first match wins)
    # R176-1a-year-fallback-fix: use fallback_text (full OCR) not raw_text (LLM output)
    if fallback_text:
        m = _YEAR_FALLBACK_RE.search(fallback_text[:METADATA_INPUT_CHAR_LIMIT])
        if m:
            return int(m.group(1))
    # Path 4: give up
    return 0


EXTRACT_PROMPT = """Extract bibliographic metadata from this document\'s first page.

Rules:
- title: Full proper title from the document header, NOT the filename. Capitalize properly.
- authors: list of "First Last" strings. Use et al. only if >5 authors and shorten.
- year: publication year as integer, e.g. 2024. Use 0 if not visible.
- doi: standardize as "10.xxxx/yyyy" without https:// prefix. Empty if not found.
- documentType: classify the document type:
  * "article" if: journal name visible, DOI in header, volume/issue numbers,
    abstract section, "Received/Accepted" dates, conference proceedings
  * "book" if: ISBN visible, "Edition" keyword, publisher name on cover,
    chapter headings (Chapter 1, Chapter 2...), no journal/DOI markers,
    table of contents on early pages
  * "thesis" if: "Thesis", "Dissertation", "PhD", "MSc", "Master\'s",
    university name + degree program, "Submitted to", "Supervisor:"
  * "unknown" if uncertain — DEFAULT when no strong signal
- isbn: ISBN-10 or ISBN-13 if visible (book/thesis). Empty for article.
- publisher: publisher name if visible (book/thesis). Empty for article.
- abstract: The paper's abstract — the summary paragraph(s) describing the work,
  usually right after the title/authors and before the Introduction. Copy verbatim
  (do NOT rewrite or translate). Empty string if there is no abstract. Do NOT
  include section headings, body text, references, or author affiliations.
- abstractVi: If the paper ALSO has a Vietnamese abstract (a section titled
  "Tóm tắt"), copy it verbatim. Empty string if there is none. Separate from
  `abstract` (the English one) — do not translate either.
- If any field truly cannot be extracted, use defaults: title="Untitled",
  authors=[], year=0, doi="", documentType="unknown", isbn="", publisher=\"\"."""


_ABSTRACT_CHAR_CAP = 3000  # R224: bound stored abstract length


class ExtractedMetadata(BaseModel):
    """Bibliographic metadata extracted from first page.

    Gemini SDK uses this Pydantic model\'s JSON schema to constrain output.

    R177-1d: extended with documentType + book fields (isbn, publisher) for
    routing in orchestrator (article→Crossref, book→Google Books).
    """

    model_config = ConfigDict(extra="ignore")

    title: str = Field(default="Untitled", description="Full proper paper title")
    authors: list[str] = Field(default_factory=list, description="Author names as First Last")
    year: int = Field(default=0, description="4-digit publication year, 0 if unknown")
    doi: str = Field(default="", description="DOI like 10.xxxx/yyyy without https://")
    # R177-1d-document-type
    document_type: str = Field(
        default="unknown",
        description="Document classification: 'article' | 'book' | 'thesis' | 'unknown'",
        alias="documentType",
    )
    isbn: str = Field(default="", description="ISBN-10 or ISBN-13 (book/thesis only)")
    publisher: str = Field(default="", description="Publisher name (book/thesis only)")
    # R222: primary language (ISO 639-1) for the en->en translation short-circuit
    language: str = Field(default="en", description="Primary language, 2-letter ISO 639-1")
    # R224: abstract verbatim — powers paper-detail panel + pre-translate source
    abstract: str = Field(default="", description="Paper abstract verbatim, empty if none")
    # R226: Vietnamese abstract (Tóm tắt) verbatim — parallel data for TM seeding
    abstract_vi: str = Field(
        default="", description="Vietnamese abstract verbatim, empty if none", alias="abstractVi"
    )


def extract_metadata(first_page_text: str) -> ExtractedMetadata:
    """Extract bibliographic metadata from OCR\'d first page.

    Best-effort. Returns defaults on any failure (TS pattern: never raises
    — metadata extraction is enhancement, not blocker for pipeline).

    Args:
        first_page_text: OCR markdown text from page 1

    Returns:
        ExtractedMetadata with whatever fields could be extracted.
        Always returns valid object — failures logged but not raised.
    """
    defaults = ExtractedMetadata()
    if not first_page_text or len(first_page_text) < MIN_INPUT_CHARS:
        return defaults

    settings = get_settings()
    text_input = first_page_text[:METADATA_INPUT_CHAR_LIMIT]

    parsed, in_tok, out_tok = extract_json(
        model=settings.gemini_model_metadata,
        prompt=text_input,
        schema=ExtractedMetadata,
        system_instruction=EXTRACT_PROMPT,
        max_tokens=settings.gemini_max_tokens_metadata,
        temperature=0.0,
        thinking_budget=0,
    )

    # Cost telemetry — best-effort log, caller doesn\'t track this step
    # Gemini 3 Flash pricing: $0.50/1M in, $3.00/1M out
    cost_usd = (in_tok * 0.50 + out_tok * 3.00) / 1_000_000
    logger.info(
        "metadata_extract_cost in_tok=%d out_tok=%d cost_usd=%.6f",
        in_tok, out_tok, cost_usd,
    )

    if parsed is None:
        logger.warning("metadata_extract_returned_none — using defaults")
        return defaults

    # Re-apply year coercion (Gemini may return string "2024" or 0)
    parsed.year = _coerce_year(parsed.year, fallback_text=first_page_text)
    parsed.language = (parsed.language or "en").strip().lower()
    parsed.abstract = (parsed.abstract or "").strip()[:_ABSTRACT_CHAR_CAP]
    parsed.abstract_vi = (parsed.abstract_vi or "").strip()[:_ABSTRACT_CHAR_CAP]

    return parsed
