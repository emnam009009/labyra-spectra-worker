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
import re
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field

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


class CrossrefReference(BaseModel):
    """One reference from a paper's Crossref-deposited reference[] list.

    Crossref reference entries are publisher-deposited (authoritative) and carry
    enough metadata to display directly — no per-DOI lookup needed. DOI-less
    entries still carry `unstructured` (the raw reference string) so they can be
    listed too.
    """

    model_config = ConfigDict(extra="forbid")

    number: int = Field(ge=1)
    doi: str | None = None
    title: str | None = None
    authors: list[str] | None = None
    year: int | None = None
    journal: str | None = None
    raw_text: str | None = None  # Crossref `unstructured` (full reference string)


def _ref_year(raw: Any) -> int | None:
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        m = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", raw)
        if m:
            return int(m.group(1))
    return None


_SUPPLEMENT_REL_KEYS = ("is-supplemented-by", "has-supplement", "has-related-material")


def extract_supplement_url(msg: dict[str, Any]) -> str | None:
    """Best-effort Supplementary Information link from a Crossref message's
    `relation` field (R237bw). Returns the first SI-style related item as a URL.

    NOTE: publishers rarely deposit SI relations (relation is mostly used for
    data/software), so this is usually empty for materials-science papers — the
    manual link (paper.siUrl) remains the primary source.
    """
    rel = msg.get("relation")
    if not isinstance(rel, dict):
        return None
    for key in _SUPPLEMENT_REL_KEYS:
        items = rel.get(key)
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            idv = it.get("id")
            if not isinstance(idv, str) or not idv.strip():
                continue
            idv = idv.strip()
            id_type = str(it.get("id-type") or "").lower()
            if id_type == "doi":
                return f"https://doi.org/{idv}"
            if id_type in ("uri", "url") or idv.startswith("http"):
                return idv
    return None


def fetch_crossref_references(doi: str) -> list[CrossrefReference]:
    """Fetch the paper's own reference list from Crossref (source A).

    Returns ordered references (incl. DOI-less ones via `unstructured`).
    Empty list when the DOI is unknown (404) or the publisher did not deposit
    references (closed/limited) — caller then falls back to PDF extraction.

    Raises:
        httpx.HTTPError: network / 5xx (caller catches).
    """
    url = f"{CROSSREF_API_BASE}/{quote(doi, safe='')}"
    headers = {"User-Agent": _user_agent(), "Accept": "application/json"}
    with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
        res = client.get(url, headers=headers)

    if res.status_code == 404:
        return []
    if res.status_code >= 400:
        raise httpx.HTTPError(f"crossref_http_{res.status_code}")

    payload = res.json()
    msg = payload.get("message") if isinstance(payload, dict) else None
    refs = msg.get("reference") if isinstance(msg, dict) else None
    if not isinstance(refs, list):
        return []

    out: list[CrossrefReference] = []
    for i, r in enumerate(refs):
        if not isinstance(r, dict):
            continue
        ref_doi = r.get("DOI")
        ref_doi = ref_doi.strip().lower() if isinstance(ref_doi, str) and ref_doi.strip() else None
        title = r.get("article-title") or r.get("volume-title") or r.get("series-title")
        title = title.strip() if isinstance(title, str) and title.strip() else None
        author = r.get("author")
        authors = [author.strip()] if isinstance(author, str) and author.strip() else None
        journal = r.get("journal-title")
        journal = journal.strip() if isinstance(journal, str) and journal.strip() else None
        raw = r.get("unstructured")
        raw = raw.strip()[:600] if isinstance(raw, str) and raw.strip() else None
        # Skip entries with no usable content at all.
        if not (ref_doi or title or raw):
            continue
        out.append(
            CrossrefReference(
                number=i + 1,
                doi=ref_doi,
                title=title,
                authors=authors,
                year=_ref_year(r.get("year")),
                journal=journal,
                raw_text=raw,
            )
        )
    return out
