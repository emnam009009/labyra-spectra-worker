"""Tests for DOI verification via Crossref/OpenAlex resolution (R237cc)."""
from __future__ import annotations

from unittest.mock import patch

import src.papers.journal_resolve as JR


class TestDoiFound:
    def test_crossref_found(self):
        with patch.object(JR, "_fetch_crossref", return_value={"container-title": ["Chem Sci"]}):
            assert JR.resolve_journal_from_doi("10.1039/x").doi_found is True

    def test_both_404_means_unverified(self):
        # OCR error like 10.1058 for 10.1038 → resolves nowhere.
        with patch.object(JR, "_fetch_crossref", return_value=None), patch.object(
            JR, "_fetch_openalex", return_value=None
        ):
            r = JR.resolve_journal_from_doi("10.1058/s43586-022-00164-0")
        assert r.doi_found is False
        assert r.source_id == ""

    def test_crossref_404_openalex_found(self):
        with patch.object(JR, "_fetch_crossref", return_value=None), patch.object(
            JR,
            "_fetch_openalex",
            return_value={"title": "X", "primary_location": {"source": {"display_name": "J"}}},
        ):
            assert JR.resolve_journal_from_doi("10.9/x").doi_found is True

    def test_crossref_work_without_journal_still_found(self):
        with patch.object(JR, "_fetch_crossref", return_value={"title": ["T"]}), patch.object(
            JR, "_fetch_openalex", return_value=None
        ):
            assert JR.resolve_journal_from_doi("10.1/x").doi_found is True

    def test_empty_doi(self):
        assert JR.resolve_journal_from_doi("").doi_found is False
