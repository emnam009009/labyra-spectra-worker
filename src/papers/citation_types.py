"""Citation entity types — Pydantic models mirroring TS contracts.

⚠ SCHEMA SYNC CONTRACT — DO NOT DRIFT ⚠

These models mirror labyra-app/src/types/citations.ts. UI in labyra-app
(R166 Phase 6b: paper detail "Cited by" + D3 network viz) reads documents
written by this worker. Field drift = silent UI breakage.

Required mirroring on any TS change:
  - Citation interface → CitationDoc class here
  - PaperCitationStats interface → PaperCitationStats class here
  - CitationConfidence union → CitationConfidence Literal here
  - metadataSource union → MetadataSource Literal here

@phase R167-B5a
@see labyra-app/src/types/citations.ts
@see docs/adr/ADR-017-citation-network.md
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ----------------------------------------------------------------------------
# Enums — must match TS union types EXACTLY
# ----------------------------------------------------------------------------
CitationConfidence = Literal["doi-exact", "title-fuzzy", "unverified", "manual"]
MetadataSource = Literal["crossref", "openalex", "pdf-only", "manual"]
CitationType = Literal["primary", "review", "methods", "background", "unknown"]
LifecycleStatus = Literal["active", "deprecated", "retracted"]


# ----------------------------------------------------------------------------
# Citation metadata returned from Crossref/OpenAlex lookup
# ----------------------------------------------------------------------------


class CitationMetadata(BaseModel):
    """Metadata returned from external DOI lookup (Crossref or OpenAlex).

    Internal worker type — NOT a Firestore document shape.
    """

    model_config = ConfigDict(extra="ignore")

    doi: str
    title: str | None = None
    authors: list[str] | None = None
    year: int | None = None
    journal: str | None = None
    is_retracted: bool = False
    source: Literal["crossref", "openalex"]


# ----------------------------------------------------------------------------
# Extracted reference (from references-parser)
# ----------------------------------------------------------------------------


class ExtractedReference(BaseModel):
    """One DOI extracted from paper text with surrounding context.

    Mirrors TS ExtractedReference interface in references-parser.ts.
    """

    model_config = ConfigDict(extra="forbid")

    doi: str
    context: str = ""


# ----------------------------------------------------------------------------
# Citation document — Firestore tenants/{tid}/citations/{id}
# ----------------------------------------------------------------------------


class CitationDoc(BaseModel):
    """Citation entity — one edge in paper citation graph.

    MUST mirror TS Citation interface (extends ProvBase). Fields ProvBase
    inherits explicitly listed here (Python has no extends).

    Firestore document shape — written by createCitation, read by labyra-app UI.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # ProvBase fields (inherited in TS, listed here)
    id: str
    schema_version: Literal[1] = Field(alias="schemaVersion", default=1)
    tenant_id: str = Field(alias="tenantId")
    created_at: int = Field(alias="createdAt")
    created_by: str = Field(alias="createdBy")
    lifecycle_status: LifecycleStatus = Field(alias="lifecycleStatus", default="active")
    derived_from: list[str] = Field(alias="derivedFrom", default_factory=list)
    generated_by: str = Field(alias="generatedBy", default="citation-extraction")

    # Citation-specific fields
    source_paper_id: str = Field(alias="sourcePaperId")
    target_doi: str | None = Field(alias="targetDoi", default=None)
    target_title: str | None = Field(alias="targetTitle", default=None)
    target_authors: list[str] | None = Field(alias="targetAuthors", default=None)
    target_year: int | None = Field(alias="targetYear", default=None)
    target_journal: str | None = Field(alias="targetJournal", default=None)
    target_paper_id: str | None = Field(alias="targetPaperId", default=None)
    metadata_source: MetadataSource | None = Field(alias="metadataSource", default=None)
    confidence: CitationConfidence
    context: str | None = None
    citation_type: CitationType | None = Field(alias="citationType", default=None)
    # R237bn (branch B): full reference listing — number in the paper's reference
    # list + the raw reference string (so DOI-less refs are still listed).
    number: int | None = Field(default=None)
    raw_text: str | None = Field(alias="rawText", default=None)


# ----------------------------------------------------------------------------
# Citation create input — worker constructs this before calling create
# ----------------------------------------------------------------------------


class CitationCreateInput(BaseModel):
    """Input shape for createCitation(). Mirrors TS CitationCreateInput
    in lib/schemas/citation-schema.ts (relevant subset).

    Worker-side type — not a Firestore shape.
    """

    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    created_by: str
    source_paper_id: str
    target_doi: str | None = None
    target_title: str | None = None
    target_authors: list[str] | None = None
    target_year: int | None = None
    target_journal: str | None = None
    target_paper_id: str | None = None
    metadata_source: MetadataSource | None = None
    confidence: CitationConfidence
    context: str | None = None
    citation_type: CitationType | None = None
    number: int | None = None
    raw_text: str | None = None


# ----------------------------------------------------------------------------
# Stats — Firestore tenants/{tid}/papers/{pid}/_stats/citations
# ----------------------------------------------------------------------------


class PaperCitationStats(BaseModel):
    """Denormalized citation counts per paper.

    Stored at: tenants/{tid}/papers/{paperId}/_stats/citations

    MUST mirror TS PaperCitationStats interface.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal[1] = Field(alias="schemaVersion", default=1)
    paper_id: str = Field(alias="paperId")
    citations_out_count: int = Field(alias="citationsOutCount", ge=0)
    citations_in_count: int = Field(alias="citationsInCount", ge=0)
    updated_at: int = Field(alias="updatedAt")
