"""Voyage AI embedding step (REST).

Port labyra-app/src/lib/ai/rag/embedding/voyage.ts + pipeline/embed-step.ts.

Model: voyage-3-large, 1024-dim, document input type.
Batch size: 128 chunks per call (matches TS EMBED_BATCH_SIZE).

Cancellation: poll Firestore between batches (mỗi 128 chunks).

@phase R167-B3
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import httpx
from google.cloud import firestore  # type: ignore[import-untyped]

from src.config import get_settings
from src.papers.errors import FatalError, RetryableError
from src.papers.pricing import voyage_embed_cost_usd
from src.papers.state import check_cancelled, increment_cost
from src.papers.types import Chunk

logger = logging.getLogger(__name__)

VOYAGE_API_BASE = "https://api.voyageai.com/v1"
VOYAGE_EMBED_MODEL = "voyage-3-large"
VOYAGE_EMBED_DIM = 1024
EMBED_BATCH_SIZE = 128
HTTP_TIMEOUT_SECONDS = 60.0


@lru_cache(maxsize=1)
def _voyage_api_key() -> str:
    """Validate + cache Voyage API key (singleton per worker instance)."""
    key = get_settings().voyage_api_key
    if not key:
        raise FatalError("VOYAGE_API_KEY missing in worker settings")
    if not key.startswith("pa-"):
        raise FatalError("VOYAGE_API_KEY malformed (expected pa-... prefix)")
    return key


class EmbeddedChunk(Chunk):
    """Chunk + embedding vector. Extends Chunk Pydantic model."""

    embedding: list[float]


def _voyage_embed_batch(texts: list[str]) -> tuple[list[list[float]], int]:
    """POST /v1/embeddings, return (embeddings, total_tokens).

    Raises:
        RetryableError: HTTP 5xx, timeout, network error
        FatalError: HTTP 4xx (bad request, auth, malformed)
    """
    if not texts:
        return [], 0

    payload: dict[str, Any] = {
        "input": texts,
        "model": VOYAGE_EMBED_MODEL,
        "input_type": "document",
    }
    headers = {
        "Authorization": f"Bearer {_voyage_api_key()}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=HTTP_TIMEOUT_SECONDS) as client:
            res = client.post(f"{VOYAGE_API_BASE}/embeddings", json=payload, headers=headers)
    except httpx.TimeoutException as exc:
        raise RetryableError(f"Voyage embed timeout: {exc}") from exc
    except httpx.HTTPError as exc:
        raise RetryableError(f"Voyage embed network error: {exc}") from exc

    if res.status_code >= 500:
        raise RetryableError(f"Voyage embed 5xx: {res.status_code} {res.text[:300]}")
    if res.status_code >= 400:
        raise FatalError(f"Voyage embed 4xx: {res.status_code} {res.text[:300]}")

    data: dict[str, Any] = res.json()
    raw = data.get("data") or []
    embeddings: list[list[float]] = []
    for item in raw:
        vec = item.get("embedding") if isinstance(item, dict) else None
        if isinstance(vec, list) and vec:
            embeddings.append(vec)

    total_tokens = int((data.get("usage") or {}).get("total_tokens") or 0)
    return embeddings, total_tokens


def run_embed_step(
    db: firestore.Client,
    tenant_id: str,
    paper_id: str,
    chunks: list[Chunk],
) -> list[EmbeddedChunk]:
    """Embed all chunks via Voyage in batches of 128.

    Mirrors TS runEmbedStep. Per batch:
      1. check_cancelled (Firestore poll)
      2. POST /embeddings
      3. validate count match
      4. increment_cost (per-paper accounting)

    Does NOT call trackUsage (quota tracker) — deferred to R168 governance
    review. Tenant-level quota stays on labyra-app side until R167-C cutover
    complete (avoids double-counting during transition).

    Raises:
        CancelledError, FatalError, RetryableError per state machine.
    """
    if not chunks:
        logger.warning("embed_skip_empty tenant=%s paper=%s", tenant_id, paper_id)
        return []

    embedded: list[EmbeddedChunk] = []

    for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
        check_cancelled(db, tenant_id, paper_id)

        batch = chunks[batch_start:batch_start + EMBED_BATCH_SIZE]
        # Use contextual_text if available (B4 enrich step), else raw text
        texts = [c.contextual_text or c.text for c in batch]

        embeddings, total_tokens = _voyage_embed_batch(texts)

        if len(embeddings) != len(batch):
            raise FatalError(
                f"embedding_count_mismatch: expected {len(batch)}, got {len(embeddings)}"
            )

        for chunk, vec in zip(batch, embeddings, strict=True):
            if len(vec) != VOYAGE_EMBED_DIM:
                raise FatalError(
                    f"embedding_dim_mismatch: expected {VOYAGE_EMBED_DIM}, got {len(vec)}"
                )
            embedded.append(EmbeddedChunk(
                chunkIdx=chunk.chunk_idx,
                text=chunk.text,
                pages=chunk.pages,
                section=chunk.section,
                tokens=chunk.tokens,
                contextualText=chunk.contextual_text,
                embedding=vec,
            ))

        cost = voyage_embed_cost_usd(total_tokens)
        increment_cost(db, tenant_id, paper_id, "embedding", cost)

        logger.info(
            "embed_batch_done tenant=%s paper=%s batch=%d size=%d tokens=%d cost=%.6f",
            tenant_id, paper_id, batch_start // EMBED_BATCH_SIZE, len(batch),
            total_tokens, cost,
        )

    logger.info(
        "embed_done tenant=%s paper=%s total_chunks=%d",
        tenant_id, paper_id, len(embedded),
    )
    return embedded
