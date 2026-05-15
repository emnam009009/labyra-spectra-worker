"""Indexing step — persist chunks to Firestore + upsert vectors to Pinecone.

Port labyra-app/src/lib/ai/rag/vector-store/pinecone.ts + pipeline/index-step.ts.

Multi-tenant pattern: single Pinecone index 'labyra-papers', one namespace
per tenant (= tenantId). Stage 1 cost optimization per Pinecone serverless
billing model (RU per GB of tenant's data, not full-index scan).

Metadata schema MUST stay in sync với TS PaperChunkMetadata interface —
this is the query-time contract for search.ts on labyra-app side. Drift =
broken retrieval. See top of file labyra-app/src/lib/ai/rag/vector-store/pinecone.ts.

@phase R167-B3
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

from google.cloud import firestore  # type: ignore[import-untyped]
from google.cloud.firestore_v1 import SERVER_TIMESTAMP  # type: ignore[import-untyped]
from pinecone import Pinecone  # type: ignore[import-untyped]

from src.config import get_settings
from src.papers.embed import EmbeddedChunk
from src.papers.errors import FatalError, RetryableError
from src.papers.state import check_cancelled
from src.papers.types import PaperDoc

logger = logging.getLogger(__name__)

PINECONE_BATCH_SIZE = 100  # Pinecone recommended max per upsert call
FIRESTORE_CHUNK_BATCH_LIMIT = 500  # Firestore batch write hard limit


@lru_cache(maxsize=1)
def _pinecone_client() -> Pinecone:
    """Singleton Pinecone client."""
    settings = get_settings()
    key = settings.pinecone_api_key
    if not key:
        raise FatalError("PINECONE_API_KEY missing in worker settings")
    if not key.startswith("pcsk_"):
        raise FatalError("PINECONE_API_KEY malformed (expected pcsk_... prefix)")
    return Pinecone(api_key=key)


def _pinecone_index() -> Any:
    """Get index handle (cheap — no network call until first op)."""
    settings = get_settings()
    return _pinecone_client().Index(settings.pinecone_index_name)


def _build_metadata(
    paper: PaperDoc,
    chunk: EmbeddedChunk,
) -> dict[str, Any]:
    """Build Pinecone metadata dict.

    Invariants (Pinecone constraints):
      - All values: str | int | float | bool | list[str]
      - No None / null
      - list[str] must be non-empty (paperAuthors fallback to ['unknown'])
      - text trimmed to 1000 chars (metadata size limit)
      - pages encoded as JSON string (Pinecone no number arrays)

    MUST match TS PaperChunkMetadata interface exactly — query side
    (labyra-app/src/lib/ai/rag/search.ts) depends on these keys.
    """
    authors = paper.authors if paper.authors else ["unknown"]
    return {
        "paperId": paper.id,
        "chunkIdx": chunk.chunk_idx,
        "text": chunk.text[:1000],
        "pagesJson": json.dumps(chunk.pages),
        "section": chunk.section,
        "paperTitle": paper.title or "Untitled",
        "paperAuthors": authors,
        "paperYear": paper.year,
        "paperDoi": paper.doi,
    }


def _firestore_write_chunks(
    db: firestore.Client,
    tenant_id: str,
    paper_id: str,
    chunks: list[EmbeddedChunk],
) -> None:
    """Write chunk docs to tenants/{tid}/papers/{pid}/chunks/{id}.

    Used by BM25 sparse retrieval (labyra-app) + citation rendering.
    Batched in groups of 500 (Firestore hard limit per batch).
    """
    for batch_start in range(0, len(chunks), FIRESTORE_CHUNK_BATCH_LIMIT):
        batch = db.batch()
        slice_ = chunks[batch_start:batch_start + FIRESTORE_CHUNK_BATCH_LIMIT]
        for chunk in slice_:
            chunk_id = f"{paper_id}-{chunk.chunk_idx}"
            ref = db.document(f"tenants/{tenant_id}/papers/{paper_id}/chunks/{chunk_id}")
            batch.set(ref, {
                "schemaVersion": 1,
                "id": chunk_id,
                "paperId": paper_id,
                "chunkIdx": chunk.chunk_idx,
                "text": chunk.text,
                "contextualText": chunk.contextual_text,
                "pages": chunk.pages,
                "section": chunk.section,
                "tokens": chunk.tokens,
                "createdAt": SERVER_TIMESTAMP,
            })
        batch.commit()


def _pinecone_upsert_batched(
    tenant_id: str,
    paper: PaperDoc,
    chunks: list[EmbeddedChunk],
) -> None:
    """Upsert vectors to Pinecone in tenant's namespace, batched at 100.

    Raises:
        FatalError: validation error (dim mismatch caught here as safety net)
        RetryableError: Pinecone API transient failure
    """
    index = _pinecone_index()

    for batch_start in range(0, len(chunks), PINECONE_BATCH_SIZE):
        slice_ = chunks[batch_start:batch_start + PINECONE_BATCH_SIZE]
        vectors = [
            {
                "id": f"{paper.id}-{c.chunk_idx}",
                "values": c.embedding,
                "metadata": _build_metadata(paper, c),
            }
            for c in slice_
        ]
        try:
            # Pinecone v5+ API: namespace via kwarg
            index.upsert(vectors=vectors, namespace=tenant_id)
        except Exception as exc:  # noqa: BLE001 — Pinecone SDK errors not strongly typed
            raise RetryableError(
                f"Pinecone upsert batch={batch_start // PINECONE_BATCH_SIZE} failed: {exc}"
            ) from exc

        logger.info(
            "pinecone_upsert_batch tenant=%s paper=%s batch=%d size=%d",
            tenant_id, paper.id, batch_start // PINECONE_BATCH_SIZE, len(slice_),
        )


def run_index_step(
    db: firestore.Client,
    tenant_id: str,
    paper: PaperDoc,
    chunks: list[EmbeddedChunk],
) -> int:
    """Run indexing: Firestore chunks + Pinecone vectors.

    Mirrors TS runIndexStep. Order:
      1. check_cancelled
      2. Firestore batch write chunks (BM25 source)
      3. check_cancelled
      4. Pinecone upsert vectors (dense retrieval)

    Returns: number of chunks indexed.

    Idempotency: re-running upserts same IDs — Pinecone overwrites,
    Firestore .set() overwrites. Safe for Pub/Sub at-least-once retries.
    """
    if not chunks:
        logger.warning("index_skip_empty tenant=%s paper=%s", tenant_id, paper.id)
        return 0

    check_cancelled(db, tenant_id, paper.id)
    _firestore_write_chunks(db, tenant_id, paper.id, chunks)
    logger.info(
        "firestore_chunks_written tenant=%s paper=%s count=%d",
        tenant_id, paper.id, len(chunks),
    )

    check_cancelled(db, tenant_id, paper.id)
    _pinecone_upsert_batched(tenant_id, paper, chunks)

    logger.info(
        "index_done tenant=%s paper=%s chunks=%d",
        tenant_id, paper.id, len(chunks),
    )
    return len(chunks)
