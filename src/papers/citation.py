"""Citation extraction step — orchestrates full extraction flow.

Port labyra-app/src/lib/ai/rag/pipeline/citation-step.ts.

Flow:
  1. extract_dois_from_text(full_text) — regex find DOIs in references section
  2. list_citations_by_source — pre-fetch existing for dedup
  3. For each DOI not yet stored:
       a. Rate-limit (200ms between calls — 5 req/s, below 50/s shared limit)
       b. lookup_doi (Crossref + OpenAlex fallback) — best-effort metadata
       c. find_internal_paper_by_doi — cross-reference to internal papers
       d. create_citation (idempotent by deterministic ID)
  4. recompute_citation_stats (denormalized counts for UI)

Non-fatal semantics:
  - Crossref/OpenAlex errors → log, count apiFailures, still create citation
    with DOI only (no metadata) so DOI relationship is preserved
  - createCitation failures → log, continue (per-DOI fault tolerance)
  - stats recompute failure → log, continue (denormalized layer is recomputable)
  - Caller (orchestrator) wraps in try/except for CancelledError propagation

@phase R167-B5b
"""
from __future__ import annotations

import logging
import time

from google.cloud import firestore  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict

from src.papers.citation_service import (
    create_citation,
    find_internal_paper_by_doi,
    list_citations_by_source,
    recompute_citation_stats,
)
from src.papers.citation_types import CitationCreateInput
from src.papers.openalex import lookup_doi
from src.papers.references_parser import extract_dois_from_text
from src.papers.state import check_cancelled

logger = logging.getLogger(__name__)

CROSSREF_RATE_LIMIT_SECONDS = 0.2  # 5 req/s — well below 50/s shared limit
MAX_DOIS_PER_PAPER = 100  # Safety cap (meta-analyses can have 500+)


class CitationStepResult(BaseModel):
    """Outcome counters for citation step. Mirrors TS CitationStepResult."""

    model_config = ConfigDict(extra="forbid")

    dois_found: int = 0
    citations_created: int = 0
    resolutions_linked: int = 0
    api_failures: int = 0


def run_citation_step(
    db: firestore.Client,
    tenant_id: str,
    paper_id: str,
    created_by: str,
    full_text: str,
) -> CitationStepResult:
    """Run citation extraction for a paper.

    Best-effort: per-DOI lookup failures are caught + logged (api_failures
    counter), but DOI relationship is still preserved (citation created
    with metadataSource='pdf-only').

    Args:
        db: Firestore client
        tenant_id: tenant
        paper_id: source paper ID
        created_by: user/system that triggered processing (for ProvBase)
        full_text: concatenated OCR text from all pages

    Returns:
        CitationStepResult with counters.

    Raises:
        CancelledError: user cancelled mid-extraction (per check_cancelled
                        between DOI lookups).
    """
    result = CitationStepResult()

    logger.info("citation_extract_start tenant=%s paper=%s", tenant_id, paper_id)

    # 1. Extract DOIs from references section
    refs = extract_dois_from_text(full_text, max_results=MAX_DOIS_PER_PAPER)
    result.dois_found = len(refs)
    logger.info(
        "citation_extract_done tenant=%s paper=%s dois_found=%d",
        tenant_id, paper_id, len(refs),
    )

    if not refs:
        return result

    # 2. Pre-fetch existing citations for dedup
    existing = list_citations_by_source(db, tenant_id, paper_id, include_deprecated=False)
    existing_dois: set[str] = {
        c.target_doi.lower() for c in existing if c.target_doi
    }

    # 3. Per-DOI lookup + create
    created_by_final = created_by or "citation-extraction-system"

    for idx, ref in enumerate(refs):
        check_cancelled(db, tenant_id, paper_id)

        # Skip already-resolved (idempotent across retries)
        if ref.doi.lower() in existing_dois:
            logger.debug("citation_skip_existing doi=%s", ref.doi)
            continue

        # Rate limit between API calls
        if idx > 0:
            time.sleep(CROSSREF_RATE_LIMIT_SECONDS)

        # Lookup metadata (best-effort, Crossref + OpenAlex fallback)
        # lookup_doi itself catches network errors and returns None
        metadata = None
        try:
            metadata = lookup_doi(ref.doi)
        except Exception as exc:  # noqa: BLE001 — defensive (lookup_doi shouldn't raise but just in case)
            result.api_failures += 1
            logger.warning(
                "citation_lookup_unexpected_error doi=%s err=%s", ref.doi, exc,
            )

        if metadata is None:
            result.api_failures += 1
            logger.info("citation_lookup_no_result doi=%s", ref.doi)

        # Cross-reference to internal paper (if cited paper also in our DB)
        try:
            internal_target = find_internal_paper_by_doi(db, tenant_id, ref.doi)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "citation_resolve_failed doi=%s err=%s", ref.doi, exc,
            )
            internal_target = None

        if internal_target:
            result.resolutions_linked += 1

        # Create citation (idempotent — deterministic ID + confidence check)
        try:
            create_citation(db, CitationCreateInput(
                tenant_id=tenant_id,
                created_by=created_by_final,
                source_paper_id=paper_id,
                target_doi=ref.doi,
                target_title=metadata.title if metadata else None,
                target_authors=metadata.authors if metadata else None,
                target_year=metadata.year if metadata else None,
                target_journal=metadata.journal if metadata else None,
                target_paper_id=internal_target,
                metadata_source=metadata.source if metadata else "pdf-only",
                confidence="doi-exact" if metadata else "unverified",  # R168-3.3
                context=ref.context,
            ))
            result.citations_created += 1
        except Exception as exc:  # noqa: BLE001 — per-DOI fault tolerance
            logger.warning(
                "citation_create_failed doi=%s err=%s", ref.doi, exc,
            )

    # 4. Recompute stats (non-fatal — denormalized layer is regenerable)
    try:
        recompute_citation_stats(db, tenant_id, paper_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "citation_stats_failed tenant=%s paper=%s err=%s",
            tenant_id, paper_id, exc,
        )

    logger.info(
        "citation_step_done tenant=%s paper=%s dois=%d created=%d linked=%d failures=%d",
        tenant_id, paper_id, result.dois_found, result.citations_created,
        result.resolutions_linked, result.api_failures,
    )
    return result
