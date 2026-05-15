"""Crossref REST API client.

Port labyra-app/src/lib/ai/citations/crossref.ts.

Free, no API key. Polite User-Agent (mailto) for higher rate share
(~50 req/s shared public pool).

Returns None on 404 (DOI not in Crossref). Raises on 5xx / network errors —
caller (citation step) catches and counts as apiFailures.

@phase R167-B5a
"""
from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import httpx

from src.config import get_settings
from src.papers.citation_types import CitationMetadata

logger = logging.getLogger(__name__)

CROSSREF_API_BASE = "https://api.crossref.org/works"
DEFAULT_TIMEOUT = 10.0  # seconds


def _polite_mailto() -> str:
    """Build polite mailto from settings."""
    settings = get_settings()
    return settings.crossref_polite_mailto or "labyra-platform@github.io"


def _user_agent() -> str:
    return f"Labyra/1.0 (mailto:{_polite_mailto()})"


def _extract_title(raw: Any) -> str | None:
    if isinstance(raw, list) and raw and isinstance(raw[0], str):
        return raw[0].strip()
    if isinstance(raw, str):
        return raw.strip()
    return None


def _extract_authors(raw: Any) -> list[str] | None:
    if not isinstance(raw, list):
        return None
    out: list[str] = []
    for a in raw:
        if not isinstance(a, dict):
            continue
        family = a.get("family")
        given = a.get("given")
        if isinstance(family, str) and family:
            if isinstance(given, str) and given:
                out.append(f"{family}, {given}")
            else:
                out.append(family)
    return out or None


def _extract_year(msg: dict[str, Any]) -> int | None:
    """Try published-print, published-online, then created."""
    for key in ("published-print", "published-online", "created"):
        block = msg.get(key)
        if isinstance(block, dict):
            date_parts = block.get("date-parts")
            if isinstance(date_parts, list) and date_parts:
                first = date_parts[0]
                if isinstance(first, list) and first and isinstance(first[0], int):
                    return first[0]
    return None


def _extract_journal(msg: dict[str, Any]) -> str | None:
    cont = msg.get("container-title")
    if isinstance(cont, list) and cont and isinstance(cont[0], str):
        return cont[0].strip()
    return None


def _is_retracted(msg: dict[str, Any]) -> bool:
    """Match TS retraction detection logic."""
    if msg.get("subtype") == "retraction" or msg.get("type") == "retraction":
        return True
    update_to = msg.get("update-to")
    if isinstance(update_to, list):
        for u in update_to:
            if isinstance(u, dict) and u.get("type") == "retraction":
                return True
    return False


def lookup_doi_crossref(doi: str) -> CitationMetadata | None:
    """Lookup paper metadata by DOI via Crossref.

    Returns:
        CitationMetadata if found, None if 404.

    Raises:
        httpx.HTTPError: network / 5xx (caller should catch + count apiFailures)
    """
    url = f"{CROSSREF_API_BASE}/{quote(doi, safe='')}"
    headers = {
        "User-Agent": _user_agent(),
        "Accept": "application/json",
    }

    with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
        res = client.get(url, headers=headers)

    if res.status_code == 404:
        return None
    if res.status_code >= 400:
        raise httpx.HTTPError(f"crossref_http_{res.status_code}")

    payload = res.json()
    if not isinstance(payload, dict):
        return None
    msg = payload.get("message")
    if not isinstance(msg, dict):
        return None

    return CitationMetadata(
        doi=doi,
        title=_extract_title(msg.get("title")),
        authors=_extract_authors(msg.get("author")),
        year=_extract_year(msg),
        journal=_extract_journal(msg),
        is_retracted=_is_retracted(msg),
        source="crossref",
    )
