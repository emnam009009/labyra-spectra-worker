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
    generate_citation_id,
    list_citations_by_source,
    recompute_citation_stats,
)
from src.papers.citation_types import CitationCreateInput
from src.papers.crossref import CrossrefReference, fetch_crossref_references
from src.papers.openalex import fetch_openalex_oa_batch, lookup_doi
from src.papers.references_agent import extract_references_with_agent
from src.papers.references_extractor import extract_references
from src.papers.state import check_cancelled
from src.papers.text_normalize import clean_text

logger = logging.getLogger(__name__)

CROSSREF_RATE_LIMIT_SECONDS = 0.2  # 5 req/s — well below 50/s shared limit
MAX_DOIS_PER_PAPER = 100  # Safety cap (meta-analyses can have 500+)


class CitationStepResult(BaseModel):
    """Outcome counters for citation step. Mirrors TS CitationStepResult."""

    model_config = ConfigDict(extra="forbid")

    dois_found: int = 0
    references_found: int = 0
    references_with_doi: int = 0
    citations_created: int = 0
    resolutions_linked: int = 0
    api_failures: int = 0


def _ingest_crossref_refs(
    db: firestore.Client,
    tenant_id: str,
    paper_id: str,
    created_by_final: str,
    refs: list[CrossrefReference],
    result: CitationStepResult,
) -> CitationStepResult:
    """Create citations from Crossref-deposited references (source A).

    Metadata (title/authors/year/journal) is already in each entry, so there is
    NO per-DOI network lookup here — only the cheap internal cross-reference
    (find_internal_paper_by_doi) for the in-library badge.
    """
    result.references_found = len(refs)
    result.references_with_doi = sum(1 for r in refs if r.doi)
    result.dois_found = result.references_with_doi

    # R237co: one batch OpenAlex call fills publisher + is_oa for all ref DOIs
    # (Crossref-deposited reference entries carry neither). Free, ~50 DOI/call.
    oa_batch: dict = {}
    ref_dois = [r.doi for r in refs if r.doi]
    if ref_dois:
        try:
            oa_batch = fetch_openalex_oa_batch(ref_dois)
        except Exception as exc:  # noqa: BLE001 — best-effort enrichment
            logger.warning("citation_oa_batch_failed paper=%s err=%s", paper_id, exc)

    for ref in refs:
        check_cancelled(db, tenant_id, paper_id)

        internal_target = None
        if ref.doi:
            try:
                internal_target = find_internal_paper_by_doi(db, tenant_id, ref.doi)
            except Exception as exc:  # noqa: BLE001
                logger.warning("citation_resolve_failed doi=%s err=%s", ref.doi, exc)
            if internal_target:
                result.resolutions_linked += 1

        oa = oa_batch.get((ref.doi or "").strip().lower()) if ref.doi else None
        try:
            create_citation(db, CitationCreateInput(
                tenant_id=tenant_id,
                created_by=created_by_final,
                source_paper_id=paper_id,
                target_doi=ref.doi,
                target_title=clean_text(ref.title) or None,
                target_authors=ref.authors,
                target_year=ref.year,
                target_journal=ref.journal,
                target_publisher=oa.publisher if oa else None,
                target_is_open_access=oa.is_oa if oa else None,
                target_paper_id=internal_target,
                metadata_source="crossref",
                confidence="doi-exact" if ref.doi else "unverified",
                context=None,
                number=ref.number,
                raw_text=ref.raw_text,
            ))
            result.citations_created += 1
        except Exception as exc:  # noqa: BLE001 — per-reference fault tolerance
            logger.warning("citation_create_failed number=%s err=%s", ref.number, exc)

    try:
        recompute_citation_stats(db, tenant_id, paper_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("citation_stats_failed tenant=%s paper=%s err=%s", tenant_id, paper_id, exc)

    logger.info(
        "citation_step_done tenant=%s paper=%s source=crossref refs=%d created=%d linked=%d",
        tenant_id, paper_id, result.references_found, result.citations_created,
        result.resolutions_linked,
    )
    return result


def run_citation_step(
    db: firestore.Client,
    tenant_id: str,
    paper_id: str,
    created_by: str,
    full_text: str,
    self_doi: str | None = None,
) -> CitationStepResult:
    """Run citation (reference list) extraction for a paper.

    Two sources (ADR-017; R237bt):
      A. Crossref reference[] of the paper itself (when self_doi is known and the
         publisher deposited references) — authoritative, ordered, carries
         title/authors/year so no per-DOI lookup is needed, and lists DOI-less
         references too. Preferred.
      B. PDF DOI-anchored extraction (extract_references) — deterministic
         fallback when A is empty (no self DOI / references closed / 404). Each
         DOI is then resolved via lookup_doi (Crossref→OpenAlex).

    Best-effort: per-DOI failures are caught + counted (api_failures); the DOI
    relationship is still preserved.

    Returns:
        CitationStepResult with counters.

    Raises:
        CancelledError: user cancelled mid-extraction.
    """
    result = CitationStepResult()
    created_by_final = created_by or "citation-extraction-system"

    logger.info("citation_extract_start tenant=%s paper=%s self_doi=%r", tenant_id, paper_id, self_doi)

    # ── Source A: the paper's own Crossref reference list ────────────────
    if self_doi:
        crossref_refs: list = []
        try:
            crossref_refs = fetch_crossref_references(self_doi)
        except Exception as exc:  # noqa: BLE001 — network/5xx; fall back to PDF
            result.api_failures += 1
            logger.warning("crossref_refs_failed doi=%s err=%s", self_doi, exc)
        if crossref_refs:
            logger.info(
                "citation_source_crossref tenant=%s paper=%s refs=%d",
                tenant_id, paper_id, len(crossref_refs),
            )
            return _ingest_crossref_refs(
                db, tenant_id, paper_id, created_by_final, crossref_refs, result
            )

    # ── Source B (fallback): PDF DOI-anchored extraction ─────────────────
    # 1. Extract the full reference list (numbered; DOI optional) — R237bn/bo.
    refs = extract_references(full_text, max_results=MAX_DOIS_PER_PAPER)
    result.references_found = len(refs)
    result.references_with_doi = sum(1 for r in refs if r.doi)
    result.dois_found = result.references_with_doi  # backward-compat field
    logger.info(
        "citation_extract_done tenant=%s paper=%s refs=%d with_doi=%d (source=pdf)",
        tenant_id, paper_id, result.references_found, result.references_with_doi,
    )

    if not refs:
        # Source C (last resort): A (Crossref) and B (PDF DOI-anchored) both
        # empty — let the agent structure the references section. DOIs returned
        # are verified verbatim inside references_agent (no fabrication).
        agent_refs, _in_tok, _out_tok = extract_references_with_agent(full_text)
        if agent_refs:
            logger.info(
                "citation_source_agent tenant=%s paper=%s refs=%d",
                tenant_id, paper_id, len(agent_refs),
            )
            refs = agent_refs
            result.references_found = len(refs)
            result.references_with_doi = sum(1 for r in refs if r.doi)
            result.dois_found = result.references_with_doi
        else:
            # R168-3.6: still create _stats doc with count=0 for UI queries
            try:
                recompute_citation_stats(db, tenant_id, paper_id)
            except Exception as exc:  # noqa: BLE001 — non-blocking, best-effort
                logger.warning(
                    "stats_recompute_failed_empty paper=%s err=%s", paper_id, exc
                )
            return result

    # 2. Pre-fetch existing citations for dedup (by deterministic ID).
    existing = list_citations_by_source(db, tenant_id, paper_id, include_deprecated=False)
    existing_ids: set[str] = {c.id for c in existing}

    # 3. Per-reference lookup + create.
    api_calls = 0

    # R237co: batch OA + publisher for all ref DOIs (one OpenAlex call, free).
    # Crossref lookup already yields publisher; OpenAlex fills is_oa (+ publisher
    # when Crossref had none).
    oa_batch: dict = {}
    ref_dois = [r.doi for r in refs if r.doi]
    if ref_dois:
        try:
            oa_batch = fetch_openalex_oa_batch(ref_dois)
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning("citation_oa_batch_failed paper=%s err=%s", paper_id, exc)

    for ref in refs:
        check_cancelled(db, tenant_id, paper_id)

        cid = generate_citation_id(paper_id, ref.doi, None, ref.raw_text)
        already = cid in existing_ids

        metadata = None
        internal_target = None
        # Only hit the network for NEW DOI references (existing ones just get
        # their number/rawText backfilled by create_citation — no API cost).
        if ref.doi and not already:
            if api_calls > 0:
                time.sleep(CROSSREF_RATE_LIMIT_SECONDS)
            api_calls += 1
            try:
                metadata = lookup_doi(ref.doi)
            except Exception as exc:  # noqa: BLE001 — defensive
                result.api_failures += 1
                logger.warning("citation_lookup_unexpected_error doi=%s err=%s", ref.doi, exc)
            if metadata is None:
                result.api_failures += 1
                logger.info("citation_lookup_no_result doi=%s", ref.doi)
            try:
                internal_target = find_internal_paper_by_doi(db, tenant_id, ref.doi)
            except Exception as exc:  # noqa: BLE001
                logger.warning("citation_resolve_failed doi=%s err=%s", ref.doi, exc)
            if internal_target:
                result.resolutions_linked += 1

        if ref.doi:
            confidence = "doi-exact" if metadata else "unverified"
        else:
            confidence = "unverified"

        oa = oa_batch.get((ref.doi or "").strip().lower()) if ref.doi else None
        try:
            create_citation(db, CitationCreateInput(
                tenant_id=tenant_id,
                created_by=created_by_final,
                source_paper_id=paper_id,
                target_doi=ref.doi,
                target_title=(clean_text(metadata.title) or None) if metadata else None,
                target_authors=metadata.authors if metadata else None,
                target_year=metadata.year if metadata else None,
                target_journal=metadata.journal if metadata else None,
                target_publisher=(
                    metadata.publisher
                    if metadata and metadata.publisher
                    else (oa.publisher if oa else None)
                ),
                target_is_open_access=(
                    metadata.is_open_access
                    if metadata and metadata.is_open_access is not None
                    else (oa.is_oa if oa else None)
                ),
                target_paper_id=internal_target,
                metadata_source=(metadata.source if metadata else ("pdf-only" if ref.doi else None)),
                confidence=confidence,
                context=None,
                number=ref.number,
                raw_text=ref.raw_text,
            ))
            result.citations_created += 1
        except Exception as exc:  # noqa: BLE001 — per-reference fault tolerance
            logger.warning("citation_create_failed id=%s err=%s", cid, exc)

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
