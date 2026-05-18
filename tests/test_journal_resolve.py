"""Tests for journal resolution (R179-2). HTTP mocked via httpx MockTransport."""
from __future__ import annotations

import httpx
import pytest

from src.papers.journal_resolve import (
    JournalResolveResult,
    _extract_issn,
    _extract_journal,
    _extract_journal_short,
    _parse_openalex,
    resolve_journal_from_doi,
)


# ─── Pydantic shape (2 tests) ────────────────────────────────────────────


def test_result_empty_defaults() -> None:
    r = JournalResolveResult()
    assert r.journal == ""
    assert r.journal_issn == []
    assert r.source_id == ""
    assert r.rejected is False


def test_result_strict_extra() -> None:
    with pytest.raises(Exception):  # ConfigDict(extra="forbid")
        JournalResolveResult(unknown_field="x")


# ─── Crossref extractors (3 tests) ───────────────────────────────────────


def test_extract_journal_array() -> None:
    msg = {"container-title": ["Journal of Materials Chemistry A"]}
    assert _extract_journal(msg) == "Journal of Materials Chemistry A"


def test_extract_journal_short() -> None:
    msg = {"short-container-title": ["J. Mater. Chem. A"]}
    assert _extract_journal_short(msg) == "J. Mater. Chem. A"


def test_extract_issn_dual() -> None:
    msg = {"ISSN": ["2050-7488", "2050-7496", "extra-ignored"]}
    assert _extract_issn(msg) == ["2050-7488", "2050-7496"]


# ─── OpenAlex parser (1 test) ────────────────────────────────────────────


def test_parse_openalex_shape() -> None:
    data = {
        "primary_location": {
            "source": {
                "display_name": "Nature Energy",
                "abbreviated_title": "Nat Energy",
                "issn_l": "2058-7546",
                "issn": ["2058-7546", "2058-7554"],
            }
        }
    }
    journal, short, issn = _parse_openalex(data)
    assert journal == "Nature Energy"
    assert short == "Nat Energy"
    assert "2058-7546" in issn
    assert len(issn) == 2


# ─── Resolver fallback paths (2 tests) ───────────────────────────────────


def test_resolve_empty_doi_rejected() -> None:
    r = resolve_journal_from_doi("")
    assert r.rejected is True
    assert r.rejected_reason == "missing_or_invalid_doi"
    assert r.source_id == ""


def test_resolve_short_doi_rejected() -> None:
    r = resolve_journal_from_doi("xx")
    assert r.rejected is True
