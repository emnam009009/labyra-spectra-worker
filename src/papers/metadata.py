"""Extract paper metadata (title, authors, year, DOI) from OCR'd first page.

Port labyra-app/src/lib/ai/rag/pipeline/metadata-extract.ts.

Uses Haiku 4.5 (~$0.001/paper). NO prompt caching — single-shot call per
paper, cache write overhead not worth it.

Returns defaults (title='Untitled', authors=[], year=0, doi='') on:
  - Empty/too-short input (<50 chars)
  - LLM call failure
  - JSON parse failure

This is best-effort metadata — citation step + downstream search still work
with defaults.

@phase R167-B4
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, Field

from src.papers.enrich import _anthropic_client  # reuse singleton

logger = logging.getLogger(__name__)

METADATA_MODEL = "claude-haiku-4-5-20251001"
METADATA_MAX_TOKENS = 500
METADATA_INPUT_CHAR_LIMIT = 4000  # match TS slice(0, 4000)
MIN_INPUT_CHARS = 50

EXTRACT_PROMPT = """Extract bibliographic metadata from this scientific paper's first page.

Return ONLY valid JSON with this exact shape, no markdown fences, no commentary:
{
  "title": "<full paper title, NO filename slugs>",
  "authors": ["<First Last>", "..."],
  "year": <4-digit year as number, 0 if unknown>,
  "doi": "<DOI like 10.1021/acsami.xxxx, empty string if not found>"
}

Rules:
- title: Full proper title from the article header, NOT the filename. Capitalize properly.
- authors: list of "First Last" strings. Use et al. only if >5 authors and shorten.
- year: publication year as integer, e.g. 2024. Use 0 if not visible.
- doi: standardize as "10.xxxx/yyyy" without https:// prefix.
- If any field truly cannot be extracted, use defaults: title="Untitled", authors=[], year=0, doi=""."""

_MARKDOWN_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)


class ExtractedMetadata(BaseModel):
    """Bibliographic metadata extracted from first page."""

    model_config = ConfigDict(extra="ignore")

    title: str = "Untitled"
    authors: list[str] = Field(default_factory=list)
    year: int = 0
    doi: str = ""


def _strip_fences(text: str) -> str:
    """Remove ```json...``` markdown wrappers if LLM added them."""
    return _MARKDOWN_FENCE_RE.sub("", text).strip()


def _parse_metadata_json(raw_text: str) -> ExtractedMetadata:
    """Parse LLM output to ExtractedMetadata, defaulting any invalid fields.

    Matches TS field-by-field defaulting (title default 'Untitled' only if
    missing/empty string).
    """
    cleaned = _strip_fences(raw_text)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("expected JSON object")

    # Defensive field handling — match TS partial typing
    title_raw = parsed.get("title")
    title = title_raw if isinstance(title_raw, str) and title_raw else "Untitled"

    authors_raw = parsed.get("authors")
    if isinstance(authors_raw, list):
        authors = [a for a in authors_raw if isinstance(a, str)]
    else:
        authors = []

    year_raw = parsed.get("year")
    year = year_raw if isinstance(year_raw, int) else 0

    doi_raw = parsed.get("doi")
    doi = doi_raw if isinstance(doi_raw, str) else ""

    return ExtractedMetadata(title=title, authors=authors, year=year, doi=doi)


def extract_metadata(first_page_text: str) -> ExtractedMetadata:
    """Extract bibliographic metadata from OCR'd first page.

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

    try:
        client = _anthropic_client()
        response = client.messages.create(
            model=METADATA_MODEL,
            max_tokens=METADATA_MAX_TOKENS,
            system=[{"type": "text", "text": EXTRACT_PROMPT}],
            messages=[{
                "role": "user",
                "content": first_page_text[:METADATA_INPUT_CHAR_LIMIT],
            }],
        )

        text_blocks = [b.text for b in response.content if hasattr(b, "text")]
        raw_text = "".join(text_blocks)

        return _parse_metadata_json(raw_text)

    except Exception as exc:  # noqa: BLE001 — best-effort, never raise
        logger.warning("metadata_extract_failed err=%s", exc)
        return defaults
