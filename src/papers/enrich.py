"""Contextual enrichment step — Anthropic Contextual Retrieval.

Port labyra-app/src/lib/ai/rag/pipeline/enrich-step.ts.

Pattern (Anthropic Contextual Retrieval, ~35% retrieval boost):
  Full document Markdown cached 1h in system prompt. Each chunk submitted
  as user message; LLM generates 50-100 token summary placing chunk in
  document context. Summary prepended to chunk text before embedding.

Cost optimization: prompt caching.
  - Chunk 0: pays full doc × $2.00/M (cache write) + chunk × $1.00/M + output × $5.00/M
  - Chunk 1..N: pays full doc × $0.10/M (cache hit) + chunk × $1.00/M + output × $5.00/M

For 100 chunks × 10k-token doc:
  Without cache: 100 × 10k × $1/M ≈ $1.00
  With cache:    (10k × $2/M) + 99 × (10k × $0.10/M) ≈ $0.12

DEFAULT OFF (ENABLE_ENRICHMENT=false). Set true to enable retrieval boost.
Mirrors TS hotfix-4 decision (cost ~$0.10/paper not justified pre-PMF).

@phase R167-B4
"""
from __future__ import annotations

import logging
from functools import lru_cache

from anthropic import Anthropic
from google.cloud import firestore  # type: ignore[import-untyped]

from src.config import get_settings
from src.papers.errors import FatalError
from src.papers.pricing import haiku_45_cost_usd
from src.papers.state import check_cancelled, increment_cost
from src.papers.types import Chunk

logger = logging.getLogger(__name__)

ENRICH_MODEL = "claude-haiku-4-5-20251001"
ENRICH_MAX_TOKENS = 150
ENRICH_TEMPERATURE = 0.3
CACHE_TTL = "1h"

ENRICH_SYSTEM_TEMPLATE = (
    "<document>\n"
    "{document}\n"
    "</document>\n\n"
    "Here is a chunk we want to situate within the whole document.\n"
    "Please give a short succinct context to situate this chunk within the "
    "overall document for the purposes of improving search retrieval of the "
    "chunk. Answer only with the succinct context and nothing else."
)


@lru_cache(maxsize=1)
def _anthropic_client() -> Anthropic:
    """Singleton Anthropic client (lru_cache for worker instance reuse)."""
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise FatalError("ANTHROPIC_API_KEY missing in worker settings")
    # max_retries bumped (default 2) so transient 429 rate-limit hits are retried
    # with backoff instead of falling back to un-enriched raw text. Combined with
    # the lowered Cloud Run max-instances, 429s should be rare.
    return Anthropic(api_key=settings.anthropic_api_key, max_retries=5)


def _enrich_one_chunk(
    full_doc_md: str,
    chunk_text: str,
) -> tuple[str, float]:
    """Call Haiku 4.5 with cached document + chunk in user message.

    Returns:
        (context_summary, cost_usd)
    """
    client = _anthropic_client()
    system_text = ENRICH_SYSTEM_TEMPLATE.format(document=full_doc_md)

    response = client.messages.create(
        model=ENRICH_MODEL,
        max_tokens=ENRICH_MAX_TOKENS,
        temperature=ENRICH_TEMPERATURE,
        system=[{
            "type": "text",
            "text": system_text,
            "cache_control": {"type": "ephemeral", "ttl": CACHE_TTL},
        }],
        messages=[{
            "role": "user",
            "content": f"<chunk>\n{chunk_text}\n</chunk>",
        }],
    )

    # Extract context summary
    text_blocks = [b.text for b in response.content if hasattr(b, "text")]
    context = "".join(text_blocks).strip()

    # Compute cost from 4 token fields (Anthropic SDK returns these separately)
    usage = response.usage
    cost = haiku_45_cost_usd(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
    )
    return context, cost


def run_enrich_step(
    db: firestore.Client,
    tenant_id: str,
    paper_id: str,
    full_document_md: str,
    chunks: list[Chunk],
) -> list[Chunk]:
    """Enrich chunks with contextual summary.

    DEFAULT OFF: when ENABLE_ENRICHMENT != 'true', returns chunks with
    contextual_text = raw text (no enrichment). Saves ~$0.10/paper.

    When ON: each chunk gets context-prepended contextual_text used by
    downstream embed step.

    Per-chunk failures fall back to raw text (matches TS pattern) — don't
    fail entire paper for one bad enrichment call. Only first failure logged
    to avoid log spam.

    Raises:
        CancelledError if user cancels mid-batch.
    """
    settings = get_settings()
    if not settings.enable_enrichment:
        logger.info(
            "enrichment_skipped tenant=%s paper=%s reason=disabled chunks=%d",
            tenant_id, paper_id, len(chunks),
        )
        # Return chunks unchanged (contextual_text already defaults to "")
        # Embed step will fall back to raw text per its own logic
        return chunks

    if not chunks:
        return []

    if not full_document_md:
        logger.warning(
            "enrichment_no_doc tenant=%s paper=%s — using raw text",
            tenant_id, paper_id,
        )
        return chunks

    enriched: list[Chunk] = []
    first_failure_logged = False
    total_cost = 0.0

    for chunk in chunks:
        check_cancelled(db, tenant_id, paper_id)

        try:
            context, cost = _enrich_one_chunk(full_document_md, chunk.text)
            contextual = f"[{context}]\n\n{chunk.text}"
            total_cost += cost
        except Exception as exc:  # noqa: BLE001 — fallback for any LLM error
            if not first_failure_logged:
                logger.warning(
                    "enrichment_failed_using_raw tenant=%s paper=%s chunk=%d err=%s",
                    tenant_id, paper_id, chunk.chunk_idx, exc,
                )
                first_failure_logged = True
            contextual = chunk.text

        # Construct new Chunk preserving all fields + new contextual_text
        enriched.append(Chunk(
            chunkIdx=chunk.chunk_idx,
            text=chunk.text,
            pages=chunk.pages,
            section=chunk.section,
            tokens=chunk.tokens,
            contextualText=contextual,
        ))

    if total_cost > 0:
        increment_cost(db, tenant_id, paper_id, "enrichment", total_cost)
        logger.info(
            "enrich_done tenant=%s paper=%s chunks=%d cost_usd=%.6f",
            tenant_id, paper_id, len(enriched), total_cost,
        )

    return enriched
