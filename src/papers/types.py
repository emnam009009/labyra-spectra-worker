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

from pydantic import BaseModel, ConfigDict, Field

# ----------------------------------------------------------------------------
# PaperStatus — must match labyra-app/src/types/papers.ts exactly
# ----------------------------------------------------------------------------
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

    # Created context (for citation step needs createdBy per R166 ai-6)
    created_by: str = Field(alias="createdBy", default="")


# ----------------------------------------------------------------------------
# Pipeline data types (Pydantic for I/O boundaries; pure dataclasses OK
# for in-process passing but Pydantic simpler — same import path)
# ----------------------------------------------------------------------------


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
