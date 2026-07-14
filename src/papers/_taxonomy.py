"""Taxonomy v1 — paper domain classification (R178-3).

36 categories across 4 axes:
  APPLICATION (13) — primary domain candidates
  MATERIALS_CLASS (9) — primary OR subtopic
  SYNTHESIS (6) — subtopic only
  CHARACTERIZATION (5) — subtopic only
  META (3) — primary only

Primary candidates = APPLICATION + MATERIALS_CLASS + META = 25 slugs.
Subtopic candidates = MATERIALS_CLASS + SYNTHESIS + CHARACTERIZATION = 20 slugs.

Constraints enforced by DomainClassification:
  - primary ∈ PRIMARY_DOMAINS
  - subtopics ⊆ SUBTOPIC_DOMAINS
  - subtopics ∌ primary (validate_no_duplicate)
  - |subtopics| ≤ 4

@phase R178-3
@r178-3-applied
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

TAXONOMY_VERSION = "v1"
PROMPT_VERSION = "v1.1"


APPLICATION_SLUGS: frozenset[str] = frozenset({
    "photocatalysis", "electrocatalysis_her", "electrocatalysis_oer",
    "solar_cells", "batteries_li_ion", "batteries_beyond_li",
    "supercapacitors", "sensors_gas", "sensors_bio",
    "water_treatment", "co2_reduction", "hydrogen_storage", "thermoelectrics",
})

MATERIALS_CLASS_SLUGS: frozenset[str] = frozenset({
    "metal_oxides", "sulfides_selenides", "mxenes", "perovskites",
    "two_d_materials", "mofs_cofs", "carbon_nanomaterials",
    "polymers_composites", "alloys_intermetallics",
})

SYNTHESIS_SLUGS: frozenset[str] = frozenset({
    "hydrothermal_solvothermal", "sol_gel", "cvd_pvd",
    "electrochemical_deposition", "mechanochemical", "green_synthesis",
})

CHARACTERIZATION_SLUGS: frozenset[str] = frozenset({
    "xrd_focused", "spectroscopy_focused", "microscopy_focused",
    "electrochemistry_focused", "dft_computational",
})

META_SLUGS: frozenset[str] = frozenset({
    "review_article", "perspective", "unknown",
})

PRIMARY_DOMAINS: frozenset[str] = APPLICATION_SLUGS | MATERIALS_CLASS_SLUGS | META_SLUGS
"""25 valid primary domain slugs."""

SUBTOPIC_DOMAINS: frozenset[str] = MATERIALS_CLASS_SLUGS | SYNTHESIS_SLUGS | CHARACTERIZATION_SLUGS
"""20 valid subtopic slugs."""

ALL_SLUGS: frozenset[str] = PRIMARY_DOMAINS | SUBTOPIC_DOMAINS
"""36 unique slugs."""


class DomainConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DomainClassification(BaseModel):
    """Gemini structured output schema. Strict enum — unknown slugs raise."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    primary: str = Field(
        description="One slug from PRIMARY_DOMAINS (25 options).",
    )
    subtopics: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="0-4 subtopic slugs (must not duplicate primary).",
    )
    confidence: DomainConfidence
    reasoning: str = Field(
        max_length=500,
        description="1-2 sentence justification citing specific terms.",
    )

    @field_validator("primary")
    @classmethod
    def _validate_primary(cls, v: str) -> str:
        if v not in PRIMARY_DOMAINS:
            raise ValueError(
                f"primary '{v}' not in PRIMARY_DOMAINS — expected one of "
                f"{sorted(PRIMARY_DOMAINS)}"
            )
        return v

    @field_validator("subtopics")
    @classmethod
    def _validate_subtopics(cls, v: list[str]) -> list[str]:
        for slug in v:
            if slug not in SUBTOPIC_DOMAINS:
                raise ValueError(
                    f"subtopic '{slug}' not in SUBTOPIC_DOMAINS — expected "
                    f"one of {sorted(SUBTOPIC_DOMAINS)}"
                )
        # Deduplicate preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for s in v:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        return deduped

    def validate_no_duplicate(self) -> "DomainClassification":
        """Cross-field: subtopics must not contain primary."""
        if self.primary in self.subtopics:
            raise ValueError(f"subtopics MUST NOT include primary '{self.primary}'")
        return self
