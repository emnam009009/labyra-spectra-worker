"""Citation Firestore CRUD service — create-side operations only.

Port labyra-app/src/lib/firebase/citations/service.ts (subset).

Path conventions (multi-tenant isolation per ADR-016):
  - Citations:   tenants/{tenantId}/citations/{id}
  - Stats:       tenants/{tenantId}/papers/{paperId}/_stats/citations

Deterministic ID generation for dedup:
  - With DOI:   {sourcePaperId}:d:{sha256(doi.lower())[:8]}
  - Title only: {sourcePaperId}:t:{sha256(title_normalized)[:8]}

Confidence ranking (existing > new = preserve existing):
  manual (3) > doi-exact (2) > title-fuzzy (1)

@phase R167-B5b
"""
from __future__ import annotations

import hashlib
import logging
import re
import time

from google.cloud import firestore  # type: ignore[import-untyped]

from src.papers.citation_types import (
    CitationConfidence,
    CitationCreateInput,
    CitationDoc,
    PaperCitationStats,
)
from src.papers.errors import FatalError

logger = logging.getLogger(__name__)

CITATIONS_COLLECTION = "citations"
STATS_SUBCOLLECTION = "_stats"
STATS_DOC = "citations"

# Confidence ranking — higher number = more trusted
_CONFIDENCE_ORDER: dict[CitationConfidence, int] = {
    "title-fuzzy": 1,
    "doi-exact": 2,
    "manual": 3,
}

_WHITESPACE_RE = re.compile(r"\s+")


def _sha256_short(s: str) -> str:
    """SHA-256 hex digest, first 8 chars. Matches TS crypto.createHash('sha256').digest('hex').slice(0,8)."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]


def generate_citation_id(
    source_paper_id: str,
    target_doi: str | None = None,
    target_title: str | None = None,
    target_raw: str | None = None,
) -> str:
    """Generate deterministic Citation ID.

    Same (source, target) → same ID → idempotent createCitation across
    Pub/Sub retries. Critical for at-least-once delivery safety.

    Precedence: DOI > title > raw reference text. The raw-text scheme (R237bn)
    lets DOI-less / un-resolved references still get a stable ID so they can be
    listed without colliding.

    Raises:
        FatalError: if target_doi, target_title and target_raw are all missing.
    """
    if target_doi:
        return f"{source_paper_id}:d:{_sha256_short(target_doi.lower())}"
    if target_title:
        normalized = _WHITESPACE_RE.sub(" ", target_title.lower()).strip()
        return f"{source_paper_id}:t:{_sha256_short(normalized)}"
    if target_raw:
        normalized = _WHITESPACE_RE.sub(" ", target_raw.lower()).strip()[:200]
        return f"{source_paper_id}:r:{_sha256_short(normalized)}"
    raise FatalError("Citation requires targetDoi, targetTitle or rawText for ID generation")


def _citations_collection(db: firestore.Client, tenant_id: str) -> firestore.CollectionReference:
    """Get citations collection ref for a tenant."""
    return db.collection("tenants").document(tenant_id).collection(CITATIONS_COLLECTION)


def _stats_doc_ref(
    db: firestore.Client, tenant_id: str, paper_id: str,
) -> firestore.DocumentReference:
    """Get stats doc ref for a paper."""
    return (
        db.collection("tenants").document(tenant_id)
        .collection("papers").document(paper_id)
        .collection(STATS_SUBCOLLECTION).document(STATS_DOC)
    )


def create_citation(db: firestore.Client, input_: CitationCreateInput) -> CitationDoc:
    """Create or update-in-place a citation edge.

    Idempotent semantics:
      - Same (source, target) → same deterministic ID
      - If existing has HIGHER OR EQUAL confidence, preserve existing
      - Otherwise overwrite with new

    Mirrors TS createCitation. Returns the citation document persisted.

    Raises:
        FatalError: if input is missing both targetDoi and targetTitle.
    """
    citation_id = generate_citation_id(
        input_.source_paper_id, input_.target_doi, input_.target_title, input_.raw_text,
    )
    ref = _citations_collection(db, input_.tenant_id).document(citation_id)
    now_ms = int(time.time() * 1000)

    # Build new Citation doc — alias-aware serialization for Firestore
    new_doc = CitationDoc(
        id=citation_id,
        schemaVersion=1,
        tenantId=input_.tenant_id,
        createdAt=now_ms,
        createdBy=input_.created_by,
        lifecycleStatus="active",
        derivedFrom=[input_.source_paper_id],
        generatedBy="citation-extraction",
        sourcePaperId=input_.source_paper_id,
        targetDoi=input_.target_doi,
        targetTitle=input_.target_title,
        targetAuthors=input_.target_authors,
        targetYear=input_.target_year,
        targetJournal=input_.target_journal,
        targetPaperId=input_.target_paper_id,
        metadataSource=input_.metadata_source,
        confidence=input_.confidence,
        context=input_.context,
        citationType=input_.citation_type,
        number=input_.number,
        rawText=input_.raw_text,
    )

    # Confidence ranking check (idempotent dedup)
    snapshot = ref.get()
    if snapshot.exists:
        existing_data = snapshot.to_dict() or {}
        existing_conf = existing_data.get("confidence")
        if existing_conf in _CONFIDENCE_ORDER:
            existing_rank = _CONFIDENCE_ORDER[existing_conf]  # type: ignore[index]
            new_rank = _CONFIDENCE_ORDER[input_.confidence]
            if existing_rank >= new_rank:
                # Existing is at least as trusted — keep its confidence/metadata,
                # but backfill the reference-listing fields (R237bn) when the
                # existing doc predates them, so a reprocess fills numbers/raw text
                # without downgrading anything.
                patch: dict = {}
                if existing_data.get("number") is None and input_.number is not None:
                    patch["number"] = input_.number
                if not existing_data.get("rawText") and input_.raw_text:
                    patch["rawText"] = input_.raw_text
                if patch:
                    ref.update(patch)
                    existing_data.update(patch)
                logger.info(
                    "citation_skip_existing id=%s existing_conf=%s new_conf=%s backfill=%s",
                    citation_id, existing_conf, input_.confidence, list(patch.keys()),
                )
                return CitationDoc.model_validate(existing_data)

    # Write (overwrite or new)
    ref.set(new_doc.model_dump(by_alias=True, exclude_none=False))
    logger.info(
        "citation_created id=%s source=%s target_doi=%s confidence=%s",
        citation_id, input_.source_paper_id, input_.target_doi, input_.confidence,
    )
    return new_doc


def list_citations_by_source(
    db: firestore.Client,
    tenant_id: str,
    source_paper_id: str,
    include_deprecated: bool = False,
) -> list[CitationDoc]:
    """List OUT-edges (this paper cites others).

    Used by citation step to skip already-resolved DOIs (dedup).
    Worker-only — UI uses TS service.listCitationsBySource.
    """
    statuses = ["active"]
    if include_deprecated:
        statuses.append("deprecated")

    query = (
        _citations_collection(db, tenant_id)
        .where("sourcePaperId", "==", source_paper_id)
        .where("lifecycleStatus", "in", statuses)
    )

    docs = []
    for snap in query.stream():
        data = snap.to_dict() or {}
        try:
            docs.append(CitationDoc.model_validate(data))
        except Exception as exc:  # noqa: BLE001 — defensive: skip malformed legacy docs
            logger.warning("citation_parse_failed id=%s err=%s", snap.id, exc)
    return docs


def find_internal_paper_by_doi(
    db: firestore.Client,
    tenant_id: str,
    doi: str,
) -> str | None:
    """Find internal paperId where Paper.doi == given DOI (cross-reference).

    Returns paperId if cited paper is also in our DB, None otherwise.
    Mirrors TS findInternalPaperByDoi in citation-step.ts.
    """
    query = (
        db.collection("tenants").document(tenant_id)
        .collection("papers")
        .where("doi", "==", doi)
        .where("lifecycleStatus", "==", "active")
        .limit(1)
    )
    for snap in query.stream():
        return snap.id
    return None


def resolve_internal_target(
    db: firestore.Client,
    tenant_id: str,
    citation_id: str,
    target_paper_id: str,
) -> None:
    """Update citation.targetPaperId when target is found in our DB.

    Mirrors TS resolveInternalTarget. Run by cross-reference background
    job after new papers are added.
    """
    _citations_collection(db, tenant_id).document(citation_id).update({
        "targetPaperId": target_paper_id,
        "updatedAt": int(time.time() * 1000),
    })


def recompute_citation_stats(
    db: firestore.Client,
    tenant_id: str,
    paper_id: str,
) -> PaperCitationStats:
    """Recompute denormalized citation counts for a paper.

    Stats stored at tenants/{tid}/papers/{pid}/_stats/citations for fast
    UI lookup. Run after Citation changes affecting this paper.

    Uses Firestore aggregation count() — single network roundtrip per side.
    """
    collection = _citations_collection(db, tenant_id)

    # OUT-edges: papers this one cites
    out_query = (
        collection
        .where("sourcePaperId", "==", paper_id)
        .where("lifecycleStatus", "==", "active")
    )
    out_count = out_query.count().get()[0][0].value

    # IN-edges: papers that cite this one
    in_query = (
        collection
        .where("targetPaperId", "==", paper_id)
        .where("lifecycleStatus", "==", "active")
    )
    in_count = in_query.count().get()[0][0].value

    stats = PaperCitationStats(
        schemaVersion=1,
        paperId=paper_id,
        citationsOutCount=out_count,
        citationsInCount=in_count,
        updatedAt=int(time.time() * 1000),
    )

    _stats_doc_ref(db, tenant_id, paper_id).set(stats.model_dump(by_alias=True))

    logger.info(
        "citation_stats tenant=%s paper=%s out=%d in=%d",
        tenant_id, paper_id, out_count, in_count,
    )
    return stats
