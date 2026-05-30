"""OpenAlex REST API client — fallback when Crossref returns 404.

Port labyra-app/src/lib/ai/citations/openalex.ts.

Free, no API key, 100k req/day public pool. Polite-pool with mailto.

Plus combined lookup_doi() function (Crossref primary, OpenAlex fallback)
matching TS lookupDoi() composite.

@phase R167-B5a
"""
from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import httpx

from src.config import get_settings
from src.papers.citation_types import CitationMetadata
from src.papers.crossref import lookup_doi_crossref

logger = logging.getLogger(__name__)

OPENALEX_API_BASE = "https://api.openalex.org/works"
DEFAULT_TIMEOUT = 10.0  # seconds


def _polite_mailto() -> str:
    settings = get_settings()
    return (
        settings.openalex_polite_mailto
        or settings.crossref_polite_mailto
        or "labyra-platform@github.io"
    )


def _extract_authors_oa(raw: Any) -> list[str] | None:
    if not isinstance(raw, list):
        return None
    out: list[str] = []
    for a in raw:
        if not isinstance(a, dict):
            continue
        author = a.get("author")
        if isinstance(author, dict):
            name = author.get("display_name")
            if isinstance(name, str) and name:
                out.append(name)
    return out or None


def _extract_journal_oa(raw: Any) -> str | None:
    if not isinstance(raw, dict):
        return None
    source = raw.get("source")
    if isinstance(source, dict):
        name = source.get("display_name")
        if isinstance(name, str):
            return name.strip()
    return None


def lookup_doi_openalex(doi: str) -> CitationMetadata | None:
    """Lookup paper metadata by DOI via OpenAlex.

    Returns:
        CitationMetadata if found, None if 404.

    Raises:
        httpx.HTTPError: network / 5xx
    """
    mailto = _polite_mailto()
    url = f"{OPENALEX_API_BASE}/doi:{quote(doi, safe='')}?mailto={quote(mailto)}"
    # OpenAlex requires an API key since 2026-02-13 (no key → 100 credits/day).
    api_key = get_settings().openalex_api_key
    if api_key:
        url += f"&api_key={quote(api_key)}"

    with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
        res = client.get(url, headers={"Accept": "application/json"})

    if res.status_code == 404:
        return None
    if res.status_code >= 400:
        raise httpx.HTTPError(f"openalex_http_{res.status_code}")

    payload = res.json()
    if not isinstance(payload, dict):
        return None

    title = payload.get("title")
    year = payload.get("publication_year")

    return CitationMetadata(
        doi=doi,
        title=title if isinstance(title, str) else None,
        authors=_extract_authors_oa(payload.get("authorships")),
        year=year if isinstance(year, int) else None,
        journal=_extract_journal_oa(payload.get("primary_location")),
        is_retracted=bool(payload.get("is_retracted")),
        source="openalex",
    )


def lookup_doi(doi: str) -> CitationMetadata | None:
    """Composite lookup: Crossref primary, OpenAlex fallback.

    Mirrors TS lookupDoi() in openalex.ts.

    Returns:
        CitationMetadata if found in either, None if both 404 / failed.

    Errors logged but not raised — citation step uses this in best-effort mode.
    """
    try:
        result = lookup_doi_crossref(doi)
        if result is not None:
            return result
    except Exception as exc:  # noqa: BLE001 — log + fallback
        logger.warning("crossref_lookup_failed doi=%s err=%s", doi, exc)

    try:
        return lookup_doi_openalex(doi)
    except Exception as exc:  # noqa: BLE001 — log + return None
        logger.warning("openalex_lookup_failed doi=%s err=%s", doi, exc)
        return None
