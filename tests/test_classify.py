"""Tests for paper domain classification (R178-3).

Coverage:
  - Taxonomy v1 invariants
  - DomainClassification Pydantic validation
  - validate_no_duplicate cross-field
  - classify_paper_domain fallback paths
  - ClassifyResult shape + audit fields

@phase R178-3
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.papers._taxonomy import (
    ALL_SLUGS,
    APPLICATION_SLUGS,
    CHARACTERIZATION_SLUGS,
    DomainClassification,
    DomainConfidence,
    MATERIALS_CLASS_SLUGS,
    META_SLUGS,
    PRIMARY_DOMAINS,
    PROMPT_VERSION,
    SUBTOPIC_DOMAINS,
    SYNTHESIS_SLUGS,
    TAXONOMY_VERSION,
)
from src.papers.classify import (
    CLASSIFY_INPUT_CHAR_LIMIT,
    MIN_INPUT_CHARS,
    ClassifyResult,
    _fallback,
    classify_paper_domain,
)


# ─── Taxonomy invariants (4) ───────────────────────────────────────────


def test_taxonomy_v1_sizes() -> None:
    assert len(APPLICATION_SLUGS) == 13
    assert len(MATERIALS_CLASS_SLUGS) == 9
    assert len(SYNTHESIS_SLUGS) == 6
    assert len(CHARACTERIZATION_SLUGS) == 5
    assert len(META_SLUGS) == 3
    assert len(ALL_SLUGS) == 36


def test_taxonomy_primary_size() -> None:
    assert len(PRIMARY_DOMAINS) == 25


def test_taxonomy_subtopic_size() -> None:
    assert len(SUBTOPIC_DOMAINS) == 20


def test_taxonomy_versions() -> None:
    assert TAXONOMY_VERSION == "v1"
    assert PROMPT_VERSION == "v1.0"


# ─── DomainClassification validation (6) ───────────────────────────────


def test_classification_valid_application_primary() -> None:
    c = DomainClassification(
        primary="photocatalysis",
        subtopics=["metal_oxides", "hydrothermal_solvothermal"],
        confidence=DomainConfidence.HIGH,
        reasoning="WO3 photocatalyst synthesized hydrothermally.",
    )
    assert c.primary == "photocatalysis"
    assert "metal_oxides" in c.subtopics


def test_classification_valid_materials_primary() -> None:
    """Materials class can be primary if no clear application."""
    c = DomainClassification(
        primary="mxenes",
        subtopics=["xrd_focused"],
        confidence=DomainConfidence.MEDIUM,
        reasoning="MXene Ti3C2 synthesis; XRD focused.",
    )
    assert c.primary == "mxenes"


def test_classification_rejects_unknown_primary() -> None:
    with pytest.raises(ValidationError):
        DomainClassification(
            primary="not_a_real_slug", subtopics=[],
            confidence=DomainConfidence.LOW, reasoning="Test.",
        )


def test_classification_rejects_synthesis_as_primary() -> None:
    """Synthesis slugs are subtopic-only."""
    with pytest.raises(ValidationError):
        DomainClassification(
            primary="sol_gel", subtopics=[],
            confidence=DomainConfidence.LOW, reasoning="Test.",
        )


def test_classification_rejects_too_many_subtopics() -> None:
    with pytest.raises(ValidationError):
        DomainClassification(
            primary="photocatalysis",
            subtopics=[
                "metal_oxides", "sulfides_selenides",
                "hydrothermal_solvothermal", "sol_gel", "xrd_focused",
            ],
            confidence=DomainConfidence.HIGH,
            reasoning="Test.",
        )


def test_classification_dedupes_subtopics() -> None:
    c = DomainClassification(
        primary="photocatalysis",
        subtopics=["metal_oxides", "xrd_focused", "metal_oxides"],
        confidence=DomainConfidence.HIGH,
        reasoning="Test dedup.",
    )
    assert c.subtopics == ["metal_oxides", "xrd_focused"]


# ─── Cross-field (2) ───────────────────────────────────────────────────


def test_validate_no_duplicate_passes_clean() -> None:
    c = DomainClassification(
        primary="photocatalysis",
        subtopics=["metal_oxides"],
        confidence=DomainConfidence.HIGH,
        reasoning="Test.",
    ).validate_no_duplicate()
    assert c.primary == "photocatalysis"


def test_validate_no_duplicate_rejects_primary_in_subtopics() -> None:
    c = DomainClassification(
        primary="metal_oxides",
        subtopics=["metal_oxides", "xrd_focused"],
        confidence=DomainConfidence.MEDIUM,
        reasoning="Test.",
    )
    with pytest.raises(ValueError, match="MUST NOT include primary"):
        c.validate_no_duplicate()


# ─── classify_paper_domain fallback (3) ────────────────────────────────


def test_classify_empty_text_returns_unknown() -> None:
    result = classify_paper_domain("")
    assert isinstance(result, ClassifyResult)
    assert result.classification.primary == "unknown"
    assert result.rejected is False
    assert result.cost_usd == 0.0


def test_classify_too_short_returns_unknown() -> None:
    result = classify_paper_domain("Hello world.")
    assert result.classification.primary == "unknown"
    assert result.input_tokens == 0


def test_classify_fallback_has_safe_defaults() -> None:
    fb = _fallback()
    assert fb.primary == "unknown"
    assert fb.subtopics == []
    assert fb.confidence == DomainConfidence.LOW


# ─── Constants sanity (1) ──────────────────────────────────────────────


def test_constants() -> None:
    assert CLASSIFY_INPUT_CHAR_LIMIT == 3000
    assert MIN_INPUT_CHARS == 50


# Total: 4+6+2+3+1 = 16 tests
