"""Resolve journal metadata from Crossref/OpenAlex by DOI (R179-2).

Best-effort: returns JournalResolveResult with empty fields if DOI missing or
lookup fails. Caller persists result + audit log unconditionally.

Pattern mirrors:
  - R178-3 classify.py (audit + best-effort)
  - R177-1 google_books.py (book resolution via external API)

@phase R179-2
@r179-2-applied
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

CROSSREF_API_BASE = "https://api.crossref.org/works"
OPENALEX_API_BASE = "https://api.openalex.org/works/doi"
DEFAULT_TIMEOUT_SEC = 10.0

# Polite mailto from env (set in deploy.sh secret CROSSREF_POLITE_MAILTO)
import os
POLITE_MAILTO = os.environ.get("CROSSREF_POLITE_MAILTO", "labyra-platform@github.io")
USER_AGENT = f"Labyra-Worker/1.0 (mailto:{POLITE_MAILTO})"


class JournalResolveResult(BaseModel):
    """Worker output: journal metadata + audit fields.

    Always returned, never raises. Empty fields = lookup failed or DOI missing.
    """

    model_config = ConfigDict(extra="forbid")

    journal: str = ""
    """Journal name (Crossref container-title)."""

    title: str = ""
    """R228: canonical article title from the publisher (Crossref/OpenAlex).
    Authoritative — preferred over the OCR/Gemini-extracted title, which can
    misread words (e.g. 'Phage' → 'Please'). Empty if lookup had no title."""

    authors: list[str] = Field(default_factory=list)
    """R237bm (gap C): canonical author list from the publisher, "Family, Given"
    (Crossref) / display name (OpenAlex). Used to overwrite misread OCR authors
    once the DOI→title override guard has accepted the record. Empty if none."""

    journal_short: str = ""
    """Short journal name (Crossref short-container-title)."""

    journal_issn: list[str] = Field(default_factory=list)
    """ISSN list (print + electronic, up to 2)."""

    source_id: str = ""
    """'crossref' | 'openalex' | '' if both failed."""

    resolved_at: int = 0
    """Epoch ms when resolution completed."""

    rejected: bool = False
    rejected_reason: str = ""


def resolve_journal_from_doi(doi: str) -> JournalResolveResult:
    """Best-effort journal lookup. Always returns result, never raises.

    Strategy: Crossref first (faster, more complete metadata). Fall back to
    OpenAlex if Crossref returns 404 or has no container-title.
    """
    import time as _time

    result = JournalResolveResult(resolved_at=int(_time.time() * 1000))

    if not doi or len(doi) < 5:
        result.rejected = True
        result.rejected_reason = "missing_or_invalid_doi"
        return result

    # Try Crossref first
    cr_data = _fetch_crossref(doi)
    if cr_data is not None:
        journal = _extract_journal(cr_data)
        journal_short = _extract_journal_short(cr_data)
        issn = _extract_issn(cr_data)
        if journal:
            result.journal = journal
            result.title = _extract_title(cr_data)
            result.authors = _extract_authors(cr_data)
            result.journal_short = journal_short
            result.journal_issn = issn
            result.source_id = "crossref"
            return result

    # Fall back to OpenAlex
    oa_data = _fetch_openalex(doi)
    if oa_data is not None:
        journal, journal_short, issn = _parse_openalex(oa_data)
        if journal:
            result.journal = journal
            oa_title = oa_data.get("title") or oa_data.get("display_name")
            if isinstance(oa_title, str):
                result.title = oa_title.strip()
            result.authors = _extract_authors_openalex(oa_data)
            result.journal_short = journal_short
            result.journal_issn = issn
            result.source_id = "openalex"
            return result

    # Both failed
    result.rejected = True
    result.rejected_reason = "no_journal_metadata_found"
    return result


def _fetch_crossref(doi: str) -> Optional[dict]:
    """Returns parsed message or None."""
    url = f"{CROSSREF_API_BASE}/{doi}"
    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT_SEC) as client:
            res = client.get(
                url,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            )
            if res.status_code == 404:
                logger.info("crossref_doi_not_found doi=%s", doi)
                return None
            res.raise_for_status()
            json_data = res.json()
            return json_data.get("message")
    except Exception as exc:
        logger.warning("crossref_lookup_failed doi=%s err=%s", doi, exc)
        return None


def _extract_journal(msg: dict) -> str:
    cont = msg.get("container-title")
    if isinstance(cont, list) and cont and isinstance(cont[0], str):
        return cont[0].strip()
    return ""


def _extract_title(msg: dict) -> str:
    """R228: Crossref article title (first non-empty entry)."""
    raw = msg.get("title")
    if isinstance(raw, list):
        for t in raw:
            if isinstance(t, str) and t.strip():
                return t.strip()
    elif isinstance(raw, str) and raw.strip():
        return raw.strip()
    return ""


def _extract_authors(msg: dict) -> list[str]:
    """R237bm: Crossref authors as "Family, Given" (gap C). Empty list if none."""
    raw = msg.get("author")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for a in raw:
        if not isinstance(a, dict):
            continue
        family = a.get("family")
        given = a.get("given")
        if isinstance(family, str) and family.strip():
            if isinstance(given, str) and given.strip():
                out.append(f"{family.strip()}, {given.strip()}")
            else:
                out.append(family.strip())
    return out


def _extract_authors_openalex(data: dict) -> list[str]:
    """R237bm: OpenAlex authorships[].author.display_name (gap C)."""
    raw = data.get("authorships")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for a in raw:
        if not isinstance(a, dict):
            continue
        author = a.get("author")
        if isinstance(author, dict):
            name = author.get("display_name")
            if isinstance(name, str) and name.strip():
                out.append(name.strip())
    return out


def _extract_journal_short(msg: dict) -> str:
    cont = msg.get("short-container-title")
    if isinstance(cont, list) and cont and isinstance(cont[0], str):
        return cont[0].strip()
    return ""


def _extract_issn(msg: dict) -> list[str]:
    """Extract up to 2 ISSN strings (print + electronic)."""
    raw = msg.get("ISSN")
    if isinstance(raw, list):
        return [s for s in raw if isinstance(s, str)][:2]
    return []


def _fetch_openalex(doi: str) -> Optional[dict]:
    """Returns OpenAlex work JSON or None."""
    url = f"{OPENALEX_API_BASE}/{doi}"
    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT_SEC) as client:
            res = client.get(
                url,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            )
            if res.status_code == 404:
                return None
            res.raise_for_status()
            return res.json()
    except Exception as exc:
        logger.warning("openalex_lookup_failed doi=%s err=%s", doi, exc)
        return None


def _parse_openalex(data: dict) -> tuple[str, str, list[str]]:
    """OpenAlex shape: primary_location.source.display_name + issn_l + issn."""
    primary = data.get("primary_location") or {}
    source = primary.get("source") or {}
    journal = (source.get("display_name") or "").strip()
    # OpenAlex: abbreviated_title for short form
    journal_short = (source.get("abbreviated_title") or "").strip()
    issn = []
    issn_l = source.get("issn_l")
    if isinstance(issn_l, str):
        issn.append(issn_l)
    raw_issn = source.get("issn")
    if isinstance(raw_issn, list):
        for s in raw_issn:
            if isinstance(s, str) and s not in issn and len(issn) < 2:
                issn.append(s)
    return journal, journal_short, issn
