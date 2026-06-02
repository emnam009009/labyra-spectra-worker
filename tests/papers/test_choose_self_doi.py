"""Tests for choose_self_doi — deterministic labelled DOI must win over the LLM
value, which can truncate (e.g. "advs"->"adv") and break doiVerified. (R238)"""
from __future__ import annotations

from src.papers.self_doi_resolver import choose_self_doi

# Realistic page-1 text carrying the canonical labelled DOI.
PAGE_WITH_DOI = (
    "Design and Synthesis of Hollow Nanostructures for Electrochemical Water Splitting\n"
    "M. Yang, C. H. Zhang, N. W. Li\n"
    "The ORCID identification number(s) can be found under "
    "https://doi.org/10.1002/advs.202105135\n"
    "DOI: 10.1002/advs.202105135\n"
    "Along with the depletion of conventional fossil fuels ..."
)


class TestChooseSelfDoi:
    def test_regex_overrides_truncated_llm_doi(self):
        # Gemini dropped the trailing 's' in "advs"; the page text is ground truth.
        doi, source = choose_self_doi("10.1002/adv.202105135", [PAGE_WITH_DOI])
        assert doi == "10.1002/advs.202105135"
        assert source == "page-text"

    def test_regex_used_even_when_llm_empty(self):
        doi, source = choose_self_doi("", [PAGE_WITH_DOI])
        assert doi == "10.1002/advs.202105135"
        assert source == "page-text"

    def test_falls_back_to_llm_when_no_labelled_doi_in_text(self):
        # Noisy/absent URL → regex finds nothing → keep the LLM value.
        doi, source = choose_self_doi("10.1021/jacs.0c01234", ["Title only, no DOI url here."])
        assert doi == "10.1021/jacs.0c01234"
        assert source == "gemini"

    def test_empty_when_neither_present(self):
        doi, source = choose_self_doi("", ["No identifiers at all."])
        assert doi == ""
        assert source == ""

    def test_agreement_keeps_doi_and_marks_page_text(self):
        # LLM and regex agree → still sourced from the deterministic text.
        doi, source = choose_self_doi("10.1002/advs.202105135", [PAGE_WITH_DOI])
        assert doi == "10.1002/advs.202105135"
        assert source == "page-text"
