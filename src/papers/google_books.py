"""Google Books API resolver for book/textbook metadata.

Used when paper documentType=\'book\' (textbooks, monographs, edited
volumes) — Crossref/OpenAlex don\'t index books reliably.

Public API:
  lookup_book_isbn(isbn) → BookMetadata | None  (exact match)
  search_book_by_title(title, authors=None) → BookMetadata | None
      (fuzzy match, Jaccard 0.8 threshold)

Best-effort module: returns None on miss, never raises. Caller decides
fallback (e.g., orchestrator may try Crossref title-search if Books miss).

Requires settings.books_api_key (mounted via Cloud Run secret in prod,
.env.local for dev). API quota: 100k/day with key.

@phase R177-1c
"""
from __future__ import annotations

import logging
import re
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from src.config import get_settings

logger = logging.getLogger(__name__)

GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"
TIMEOUT_SECONDS = 10.0
TITLE_JACCARD_THRESHOLD = 0.8  # match labyra-app/scripts/_backfill-crossref-metadata.mjs

# ISBN validation regex (10 or 13 digits, optionally with hyphens/spaces)
_ISBN10_RE = re.compile(r"^[\d\-\s]{9,13}[\dXx]$")
_ISBN13_RE = re.compile(r"^[\d\-\s]{13,17}$")


class BookMetadata(BaseModel):
    """Bibliographic metadata for a book/textbook from Google Books.

    Field shapes mirror CitationMetadata from citation_types.py for
    downstream merge compatibility, with book-specific additions.
    """

    model_config = ConfigDict(extra="ignore")

    # Core fields (parallel to article CitationMetadata)
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int = 0
    publisher: str = ""

    # Book-specific fields
    isbn_10: str = ""
    isbn_13: str = ""
    page_count: int = 0
    subtitle: str = ""
    description: str = ""

    # Source tracing (which API + which book ID)
    source: str = "google-books"
    source_id: str = ""  # Google Books volume ID


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_isbn(isbn: str) -> str:
    """Strip hyphens, spaces from ISBN. Uppercase X for ISBN-10 checksum."""
    return re.sub(r"[\s\-]", "", isbn).upper()


def jaccard_similarity(s1: str, s2: str) -> float:
    """Token-set Jaccard similarity (case-insensitive, alphanumeric tokens only).

    Returns 0.0 to 1.0. Used to validate fuzzy title matches against the
    OCR-extracted title before accepting an API result.

    Examples:
        >>> jaccard_similarity("Infrared and Raman Spectroscopy",
        ...                    "Infrared and Raman Spectroscopy: Methods and Applications")
        0.6
    """
    tokens1 = set(re.findall(r"\w+", s1.lower()))
    tokens2 = set(re.findall(r"\w+", s2.lower()))
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union)


def _extract_volume_info(item: dict[str, Any]) -> BookMetadata | None:
    """Map a Google Books volumes API item → BookMetadata.

    Returns None if title missing (unusable result).
    """
    vol = item.get("volumeInfo", {})
    title = vol.get("title", "").strip()
    if not title:
        return None

    # Extract year from publishedDate (formats: "1995", "1995-03", "1995-03-07")
    published = vol.get("publishedDate", "")
    year = 0
    if published:
        m = re.match(r"^(\d{4})", published)
        if m:
            year = int(m.group(1))

    # Extract ISBNs
    isbn_10 = ""
    isbn_13 = ""
    for ident in vol.get("industryIdentifiers", []) or []:
        if ident.get("type") == "ISBN_10":
            isbn_10 = ident.get("identifier", "")
        elif ident.get("type") == "ISBN_13":
            isbn_13 = ident.get("identifier", "")

    return BookMetadata(
        title=title,
        subtitle=vol.get("subtitle", "").strip(),
        authors=[a.strip() for a in vol.get("authors", []) if isinstance(a, str)],
        year=year,
        publisher=vol.get("publisher", "").strip(),
        isbn_10=isbn_10,
        isbn_13=isbn_13,
        page_count=int(vol.get("pageCount", 0) or 0),
        description=vol.get("description", "").strip(),
        source="google-books",
        source_id=item.get("id", ""),
    )


def _query_google_books(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Low-level GET to volumes endpoint. Returns items list (possibly empty).

    Never raises — logs errors and returns []. Callers branch on empty.
    """
    settings = get_settings()
    if not settings.books_api_key:
        logger.warning("books_api_key_missing — Google Books lookup disabled")
        return []

    params = {
        "q": query,
        "maxResults": max_results,
        "key": settings.books_api_key,
        # Restrict to BOOK printType (excludes magazines, journals)
        "printType": "books",
    }

    try:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
            response = client.get(GOOGLE_BOOKS_URL, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("items", []) or []
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "google_books_http_error status=%d query=%s",
            exc.response.status_code,
            query[:80],
        )
        return []
    except httpx.HTTPError as exc:
        logger.warning("google_books_network_error err=%s query=%s", exc, query[:80])
        return []
    except Exception as exc:  # noqa: BLE001 — best-effort
        logger.warning("google_books_unknown_error err=%s query=%s", exc, query[:80])
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def lookup_book_isbn(isbn: str) -> BookMetadata | None:
    """Look up book by ISBN (10 or 13 digits, hyphens OK).

    Highest-confidence resolution path. ISBN is unique → first hit accepted.

    Args:
        isbn: ISBN-10 or ISBN-13, with or without hyphens.

    Returns:
        BookMetadata on hit, None on miss / network error / invalid ISBN.
    """
    if not isbn:
        return None

    normalized = _normalize_isbn(isbn)
    if not (_ISBN10_RE.match(normalized) or _ISBN13_RE.match(normalized)):
        logger.debug("invalid_isbn_format isbn=%s", isbn)
        return None

    items = _query_google_books(f"isbn:{normalized}", max_results=1)
    if not items:
        return None

    return _extract_volume_info(items[0])


def search_book_by_title(
    title: str,
    authors: list[str] | None = None,
) -> BookMetadata | None:
    """Search book by title (+ optional first author for disambiguation).

    Validates result via Jaccard 0.8 to reject false-positive matches —
    Google Books search is generous and can return unrelated books for
    short or generic titles.

    Args:
        title: book title from OCR / metadata extract
        authors: list of author names (only first author used in query)

    Returns:
        BookMetadata on hit AND Jaccard ≥ 0.8, None otherwise.
    """
    if not title or len(title) < 5:
        return None

    # R177-1c: reject 1-2 token generic titles ("Physics", "Chemistry")
    # which match too easily in Google Books. Academic papers/books rarely
    # have <3 token titles, so this filters false positives without losing
    # real cases.
    tokens = re.findall(r"\w+", title)
    if len(tokens) < 3:
        return None

    # Build query: intitle for precision, inauthor for disambiguation
    query_parts = [f'intitle:"{title}"']
    if authors and len(authors) > 0:
        first_author = authors[0].strip()
        if first_author:
            query_parts.append(f'inauthor:"{first_author}"')
    query = " ".join(query_parts)

    items = _query_google_books(query, max_results=5)
    if not items:
        return None

    # Find best Jaccard match above threshold
    best: BookMetadata | None = None
    best_score = 0.0
    for item in items:
        candidate = _extract_volume_info(item)
        if candidate is None:
            continue
        score = jaccard_similarity(title, candidate.title)
        if score > best_score:
            best_score = score
            best = candidate

    if best_score < TITLE_JACCARD_THRESHOLD:
        logger.info(
            "google_books_low_confidence best_score=%.2f title=%s",
            best_score,
            title[:60],
        )
        return None

    return best
