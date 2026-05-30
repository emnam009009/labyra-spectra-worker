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
from pydantic import BaseModel, ConfigDict, Field

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


class OpenAlexTopic(BaseModel):
    """Authoritative classification from OpenAlex's `primary_topic` (R237bz).

    OpenAlex assigns every work a 4-level path (Domain → Field → Subfield →
    Topic) via an ML model trained on title/abstract/citations/journal, in
    collaboration with CWTS Leiden. More trustworthy than a single Gemini
    guess. All fields default empty so a partial payload never raises.
    """

    model_config = ConfigDict(populate_by_name=True)

    topic_id: str = Field(default="", alias="topicId")
    topic: str = ""
    subfield: str = ""
    field: str = ""
    domain: str = ""
    score: float = 0.0


def _oa_name(block: Any) -> str:
    """display_name of a {id, display_name} sub-object, '' if absent."""
    if isinstance(block, dict):
        name = block.get("display_name")
        if isinstance(name, str):
            return name.strip()
    return ""


def fetch_openalex_topic(doi: str) -> OpenAlexTopic | None:
    """Look up a work by DOI and return its primary_topic path.

    Best-effort: returns None on 404 / any error so classification stays
    non-blocking. Looking up a single work by DOI is FREE on OpenAlex
    (singleton endpoint), but a key is still required since 2026-02-13.
    """
    clean = (doi or "").strip()
    if not clean:
        return None
    settings = get_settings()
    url = (
        f"{OPENALEX_API_BASE}/doi:{quote(clean, safe='')}"
        f"?select=primary_topic&mailto={quote(_polite_mailto())}"
    )
    if settings.openalex_api_key:
        url += f"&api_key={quote(settings.openalex_api_key)}"
    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
            res = client.get(url, headers={"Accept": "application/json"})
    except httpx.HTTPError as exc:
        logger.warning("openalex_topic_fetch_error doi=%s err=%s", clean, exc)
        return None
    if res.status_code == 404:
        return None
    if res.status_code >= 400:
        logger.warning("openalex_topic_http doi=%s status=%s", clean, res.status_code)
        return None
    try:
        payload = res.json()
    except ValueError:
        return None
    pt = payload.get("primary_topic") if isinstance(payload, dict) else None
    if not isinstance(pt, dict):
        return None
    score = pt.get("score")
    return OpenAlexTopic(
        topicId=str(pt.get("id") or "").strip(),
        topic=str(pt.get("display_name") or "").strip(),
        subfield=_oa_name(pt.get("subfield")),
        field=_oa_name(pt.get("field")),
        domain=_oa_name(pt.get("domain")),
        score=float(score) if isinstance(score, (int, float)) else 0.0,
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


def _extract_publisher_oa(primary_location: Any) -> str | None:
    """Publisher from primary_location.source.host_organization_name (R237co)."""
    if not isinstance(primary_location, dict):
        return None
    source = primary_location.get("source")
    if isinstance(source, dict):
        name = source.get("host_organization_name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return None


def _extract_is_oa(open_access: Any) -> bool | None:
    """open_access.is_oa (R237co). None when the block is absent."""
    if isinstance(open_access, dict) and isinstance(open_access.get("is_oa"), bool):
        return open_access["is_oa"]
    return None


class OaInfo(BaseModel):
    """Publisher + Open-Access flag for one DOI (R237co batch enrich)."""

    model_config = ConfigDict(extra="ignore")

    publisher: str | None = None
    is_oa: bool | None = None


def _normalize_oa_doi(raw: Any) -> str:
    """OpenAlex returns doi as a full URL; reduce to bare lower-case DOI."""
    if not isinstance(raw, str):
        return ""
    return raw.replace("https://doi.org/", "").replace("http://doi.org/", "").strip().lower()


def fetch_openalex_oa_batch(dois: list[str]) -> dict[str, OaInfo]:
    """Batch-fetch publisher + is_oa for many DOIs (R237co).

    OpenAlex supports `filter=doi:A|B|C` (OR), so ~50 references resolve in one
    request. DOI lookups are free + don't consume the daily credit cap. Keyed by
    bare lower-case DOI. Best-effort: missing/failed chunks are simply absent.
    """
    clean = [d.strip() for d in dois if d and d.strip()]
    if not clean:
        return {}

    mailto = _polite_mailto()
    api_key = get_settings().openalex_api_key
    out: dict[str, OaInfo] = {}
    chunk_size = 50
    for start in range(0, len(clean), chunk_size):
        chunk = clean[start : start + chunk_size]
        filter_val = "|".join(chunk)
        url = (
            f"{OPENALEX_API_BASE}?filter=doi:{quote(filter_val, safe='|')}"
            f"&select=doi,open_access,primary_location&per-page={chunk_size}"
            f"&mailto={quote(mailto)}"
        )
        if api_key:
            url += f"&api_key={quote(api_key)}"
        try:
            with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
                res = client.get(url, headers={"Accept": "application/json"})
        except httpx.HTTPError as exc:
            logger.warning("openalex_oa_batch_error err=%s", exc)
            continue
        if res.status_code != 200:
            logger.warning("openalex_oa_batch_http status=%s", res.status_code)
            continue
        try:
            results = res.json().get("results", [])
        except ValueError:
            continue
        if not isinstance(results, list):
            continue
        for work in results:
            if not isinstance(work, dict):
                continue
            key = _normalize_oa_doi(work.get("doi"))
            if not key:
                continue
            out[key] = OaInfo(
                publisher=_extract_publisher_oa(work.get("primary_location")),
                is_oa=_extract_is_oa(work.get("open_access")),
            )
    return out


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
    primary_location = payload.get("primary_location")

    return CitationMetadata(
        doi=doi,
        title=title if isinstance(title, str) else None,
        authors=_extract_authors_oa(payload.get("authorships")),
        year=year if isinstance(year, int) else None,
        journal=_extract_journal_oa(primary_location),
        publisher=_extract_publisher_oa(primary_location),
        is_open_access=_extract_is_oa(payload.get("open_access")),
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
