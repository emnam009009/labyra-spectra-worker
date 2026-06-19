"""Pydantic models for paper processing.

Source of truth (TS): labyra-app/src/types/papers.ts (PROV-O extended,
schemaVersion=2 per ADR-016). This file must stay in sync — schema drift
between TS publisher (labyra-app) and Python subscriber (worker) is the
#1 production risk for Pub/Sub-based decoupling.

Schema sync policy:
  - When PaperStatus enum changes in TS → update PaperStatus Literal here SAME PR
  - When Paper interface adds field → add to PaperDoc here SAME PR (unless intentionally
    not needed by worker — document with comment)

@phase R167-B1
"""
from __future__ import annotations

from typing import Literal

from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field, field_validator

# ----------------------------------------------------------------------------
# PaperStatus — must match labyra-app/src/types/papers.ts exactly
# ----------------------------------------------------------------------------
# R177-1d: documentType for routing metadata resolution
# (article → Crossref+OpenAlex DOI lookup; book → Google Books ISBN/title)
DocumentType = Literal["article", "book", "thesis", "unknown"]


PaperStatus = Literal[
    "queued",
    "ocr",
    "chunking",
    "enriching",
    "embedding",
    "indexing",
    "extracting_citations",
    "indexed",
    "failed",
    "cancelling",
    "cancelled",
]

TERMINAL_STATUSES: frozenset[PaperStatus] = frozenset({"indexed", "failed", "cancelled"})

CANCELLABLE_STATUSES: frozenset[PaperStatus] = frozenset(
    {"queued", "ocr", "chunking", "enriching", "embedding", "indexing"}
)

# ----------------------------------------------------------------------------
# Pub/Sub envelope (ADR-018 message shape)
# ----------------------------------------------------------------------------



# ----------------------------------------------------------------------------
# Timestamp coercion validators (R176-1c-1)
# ----------------------------------------------------------------------------
# See state.py parse_epoch_ms() docstring for full rationale.
# Pydantic v2 field_validator with mode='before' runs BEFORE type coercion,
# allowing us to coerce Firestore Timestamp objects → int epoch ms.
# R176-1c-1-timestamp-defensive


def _coerce_timestamp_to_epoch_ms(v: object) -> int:
    """Pydantic before-validator: Firestore timestamp → int epoch ms."""
    if v is None or v == 0 or v == "":
        return 0
    if isinstance(v, bool):
        return 0
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, datetime):
        return int(v.timestamp() * 1000)
    if isinstance(v, str):
        s = v.strip()
        if s.isdigit():
            return int(s)
    # Unknown — return 0 (validator never raises, defensive)
    return 0


class PaperJob(BaseModel):
    """Pub/Sub message body cho paper-processing topic.

    Published bởi labyra-app /api/papers/upload (R167-C cutover).
    Verified by Cloud Run IAM (--push-auth-service-account binding).

    Note vs TS jobs/types.ts PaperProcessingJob: TS thiếu storagePath +
    createdBy vì Stage 1 in-process queue share Firestore. Stage 2 worker
    nhận đủ 6 fields để tránh extra Firestore read per job.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    job_id: str = Field(alias="jobId", min_length=1)
    tenant_id: str = Field(alias="tenantId", min_length=1)
    paper_id: str = Field(alias="paperId", min_length=1)
    version: int = Field(ge=1)
    storage_path: str = Field(alias="storagePath", min_length=1)
    created_by: str = Field(alias="createdBy", min_length=1)
    enqueued_at: int = Field(alias="enqueuedAt", ge=0)


# ----------------------------------------------------------------------------
# Cost tracking (per Paper.costUsd Firestore field)
# ----------------------------------------------------------------------------


class PaperCost(BaseModel):
    """Cost breakdown per paper, matches Firestore Paper.costUsd structure."""

    model_config = ConfigDict(extra="ignore")

    ocr: float = 0.0
    enrichment: float = 0.0
    embedding: float = 0.0
    total: float = 0.0


# ----------------------------------------------------------------------------
# PaperDoc — partial read model for Firestore tenants/{tid}/papers/{id}
# ----------------------------------------------------------------------------
# Worker chỉ đọc fields cần thiết cho pipeline orchestration. Không model
# toàn bộ Paper interface (PROV-O fields, versioning fields) — đó là
# concern của labyra-app. extra='ignore' để forward-compatible khi TS
# thêm field mới mà worker chưa cần.


class PaperDoc(BaseModel):
    """Subset of Paper used by worker pipeline.

    extra='ignore' intentional — labyra-app có nhiều fields PROV-O mà worker
    không touch. Worker chỉ care về processing state machine.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    tenant_id: str = Field(alias="tenantId")
    storage_path: str = Field(alias="storagePath")
    status: PaperStatus
    retry_count: int = Field(alias="retryCount", default=0)
    max_retries: int = Field(alias="maxRetries", default=3)
    cancel_requested_at: int = Field(alias="cancelRequestedAt", default=0)

    # Metadata fields (filled during processing by metadata-extract step)
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int = 0
    doi: str = ""
    # R282: DOI provenance. labyra-app sets "manual" when the user corrects
    # the DOI; the orchestrator then preserves it instead of re-extracting.
    doi_source: str = Field(default="", alias="doiSource")

    # ADR-034 TEAM-5: research group for KB isolation. Read from the paper doc
    # (set by labyra-app on upload + backfilled). Default 'lab-shared' matches
    # the Firestore-rules defensive default — a missing field = visible to all.
    group_id: str = Field(alias="groupId", default="lab-shared")

    # R177-1d: book/non-article document type support
    # documentType detected by metadata.py Gemini extract; routes resolution
    # path in orchestrator (article=Crossref+OpenAlex, book=Google Books).
    # Default "unknown" preserves backward-compat for legacy papers.
    document_type: DocumentType = Field(alias="documentType", default="unknown")
    isbn: str = ""
    publisher: str = ""

    # Created context (for citation step needs createdBy per R166 ai-6)
    created_by: str = Field(alias="createdBy", default="")


# ----------------------------------------------------------------------------
# Pipeline data types (Pydantic for I/O boundaries; pure dataclasses OK
# for in-process passing but Pydantic simpler — same import path)
# ----------------------------------------------------------------------------

    @field_validator("cancel_requested_at", mode='before')
    @classmethod
    def _coerce_paperdoc_timestamps(cls, v: object) -> int:
        # R176-1c-1-timestamp-defensive
        return _coerce_timestamp_to_epoch_ms(v)



    # R178-3: domain classification (taxonomy v1) — @r178-3-hotfix2-applied
    domain: str = Field(default="", alias="domain")
    """Primary domain slug (one of PRIMARY_DOMAINS, 25 options)."""

    subtopics: list[str] = Field(default_factory=list, alias="subtopics")
    """0-4 subtopic slugs from SUBTOPIC_DOMAINS."""

    domain_confidence: str = Field(default="", alias="domainConfidence")
    """high|medium|low confidence label from Gemini."""

    domain_classified_at: int = Field(default=0, alias="domainClassifiedAt")
    """Epoch ms when classification was written."""

    domain_model_version: str = Field(default="", alias="domainModelVersion")
    """e.g., gemini-3-flash-preview — for audit + reclassify targeting."""

    domain_prompt_version: str = Field(default="", alias="domainPromptVersion")
    """e.g., v1.0 — bump when prompt changes."""

    domain_taxonomy_version: str = Field(default="", alias="domainTaxonomyVersion")
    """e.g., v1 — bump when taxonomy slugs change."""

    # R179-2: journal metadata via Crossref/OpenAlex — @r179-2-applied
    journal: str = Field(default="", alias="journal")
    """Full journal name from Crossref container-title."""

    journal_short: str = Field(default="", alias="journalShort")
    """Abbreviated journal name (Crossref short-container-title)."""

    journal_issn: list[str] = Field(default_factory=list, alias="journalIssn")
    """0-2 ISSN strings (print + electronic)."""

    journal_source_id: str = Field(default="", alias="journalSourceId")
    """'crossref' | 'openalex' | '' if both failed."""

    # R237bm: self-DOI resolution (Phase 1) — @doi-resolution-v1
    self_doi_source: str = Field(default="", alias="selfDoiSource")
    """How the paper's OWN DOI was found: '' (none) | 'gemini' (page-1 extract)
    | 'page-text' (labelled DOI on pages 1-3) | 'crossref-title' (Phase 2)."""

    doi_title_mismatch: bool = Field(default=False, alias="doiTitleMismatch")
    """True when a resolved-by-DOI title was WITHHELD because it didn't match the
    OCR title/authors (guard A). The DOI is kept; the OCR title is preserved for
    manual confirmation. Surfaces in the UI as a 'check title' hint."""

    journal_resolved_at: int = Field(default=0, alias="journalResolvedAt")
    """Epoch ms when Step 1e completed."""


class OcrPage(BaseModel):
    """Single page OCR result."""

    model_config = ConfigDict(extra="ignore")

    page_number: int = Field(alias="pageNumber", ge=1)
    text: str = ""


class OcrResult(BaseModel):
    """Full OCR result from Mistral provider."""

    model_config = ConfigDict(extra="ignore")

    full_text: str = Field(alias="fullText", default="")
    pages: list[OcrPage] = Field(default_factory=list)
    page_count: int = Field(alias="pageCount", ge=0)
    cost_usd: float = Field(alias="costUsd", ge=0.0)


class Chunk(BaseModel):
    """Single chunk after splitting OCR text."""

    model_config = ConfigDict(extra="forbid")

    chunk_idx: int = Field(alias="chunkIdx", ge=0)
    text: str = Field(min_length=1)
    pages: list[int] = Field(default_factory=list)
    section: str = ""
    tokens: int = Field(ge=0)
    contextual_text: str = Field(alias="contextualText", default="")
