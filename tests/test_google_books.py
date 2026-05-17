"""Unit tests for src/papers/google_books.py (R177-1c).

Mix of pure-function unit tests (no network) + real API integration tests
that skip when GEMINI/BOOKS keys missing. Matches existing worker pattern
(see test_cod_client_integration.py).

@phase R177-1f
"""
from __future__ import annotations

import os
import pytest

from src.papers.google_books import (
    BookMetadata,
    _normalize_isbn,
    jaccard_similarity,
    lookup_book_isbn,
    search_book_by_title,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pure-function unit tests (no network)
# ─────────────────────────────────────────────────────────────────────────────


class TestJaccardSimilarity:
    def test_identical_strings(self) -> None:
        assert jaccard_similarity("Infrared Spectroscopy", "Infrared Spectroscopy") == 1.0

    def test_disjoint_strings(self) -> None:
        assert jaccard_similarity("apple banana", "carrot dragon") == 0.0

    def test_partial_overlap(self) -> None:
        # 3 common tokens out of 5 total unique → 3/5 = 0.6
        score = jaccard_similarity(
            "Infrared and Raman Spectroscopy",
            "Infrared and Raman Spectroscopy Methods Applications",
        )
        assert 0.6 < score < 0.7, f"expected ~0.67, got {score}"

    def test_case_insensitive(self) -> None:
        assert jaccard_similarity("HELLO World", "hello world") == 1.0

    def test_empty_strings(self) -> None:
        assert jaccard_similarity("", "") == 0.0
        assert jaccard_similarity("hello", "") == 0.0

    def test_punctuation_ignored(self) -> None:
        # Punctuation is non-\w so split as token boundary
        score = jaccard_similarity("hello, world!", "hello world")
        assert score == 1.0


class TestNormalizeIsbn:
    def test_strip_hyphens(self) -> None:
        assert _normalize_isbn("3-527-26446-9") == "3527264469"

    def test_strip_spaces(self) -> None:
        assert _normalize_isbn("978 3527 264469") == "9783527264469"

    def test_uppercase_x_checksum(self) -> None:
        assert _normalize_isbn("0-306-40615-x") == "030640615X"

    def test_already_normalized(self) -> None:
        assert _normalize_isbn("9783527264469") == "9783527264469"


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests (real Google Books API — skip if no key)
# ─────────────────────────────────────────────────────────────────────────────

_HAS_BOOKS_KEY = bool(os.environ.get("BOOKS_API_KEY"))
_REQUIRES_BOOKS = pytest.mark.skipif(
    not _HAS_BOOKS_KEY, reason="BOOKS_API_KEY not set"
)


@_REQUIRES_BOOKS
class TestLookupBookIsbn:
    """Integration: real Google Books API with R176-1f Schrader textbook."""

    SCHRADER_ISBN = "3527264469"
    SCHRADER_TITLE = "Infrared and Raman Spectroscopy"
    SCHRADER_AUTHOR = "Bernhard Schrader"

    def test_lookup_isbn_returns_metadata(self) -> None:
        result = lookup_book_isbn(self.SCHRADER_ISBN)
        assert result is not None
        assert isinstance(result, BookMetadata)
        assert "Infrared" in result.title
        assert "Raman" in result.title
        assert self.SCHRADER_AUTHOR in result.authors
        assert result.year == 1995
        assert "Wiley" in result.publisher
        assert result.isbn_10 == self.SCHRADER_ISBN
        assert result.page_count > 0

    def test_lookup_isbn_with_hyphens(self) -> None:
        result = lookup_book_isbn("3-527-26446-9")
        assert result is not None
        assert result.isbn_10 == "3527264469"

    def test_lookup_invalid_isbn(self) -> None:
        assert lookup_book_isbn("not-an-isbn") is None
        assert lookup_book_isbn("") is None
        assert lookup_book_isbn("12345") is None  # too short

    def test_lookup_nonexistent_isbn(self) -> None:
        # Well-formed but no book exists
        result = lookup_book_isbn("9999999999999")
        assert result is None


@_REQUIRES_BOOKS
class TestSearchBookByTitle:
    """Integration: real Google Books title search + Jaccard validation."""

    def test_title_search_accepts_above_threshold(self) -> None:
        result = search_book_by_title(
            "Infrared and Raman Spectroscopy",
            authors=["Bernhard Schrader"],
        )
        assert result is not None
        assert "Infrared" in result.title

    def test_title_search_rejects_short_title(self) -> None:
        # Min 5 char + min 3 tokens
        assert search_book_by_title("ABC") is None
        assert search_book_by_title("") is None

    def test_title_search_rejects_generic_title(self) -> None:
        # 1-2 token titles like "Physics" / "Quantum Mechanics" rejected
        # before query (anti-false-positive)
        assert search_book_by_title("Physics") is None
        assert search_book_by_title("Quantum Mechanics") is None

    def test_title_search_unrelated_topic(self) -> None:
        # Article-style title — Google Books shouldn\'t have it
        result = search_book_by_title(
            "Tungsten Trioxide WO3 Nanostructures Photocatalytic Activity",
            authors=["Nguyen Van A"],
        )
        # Either None (no match) or low Jaccard (rejected) — both acceptable
        assert result is None or jaccard_similarity(
            "Tungsten Trioxide WO3 Nanostructures Photocatalytic Activity",
            result.title,
        ) >= 0.8
