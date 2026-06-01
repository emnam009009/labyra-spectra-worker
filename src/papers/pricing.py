"""Vendor pricing constants + cost functions.

Single source of truth cho cost accounting. Khi vendor đổi giá, update HERE.
Caller MUST go through these functions — không tính cost inline.

@phase R167-B2
"""
from __future__ import annotations

# ----------------------------------------------------------------------------
# Mistral OCR — $1 per 1000 pages (batch rate, per Mistral 2026 pricing)
# Source: docs Mistral OCR, confirmed labyra-app/src/lib/ai/rag/ocr/mistral.ts
# ----------------------------------------------------------------------------
MISTRAL_OCR_USD_PER_1000_PAGES: float = 1.0


def mistral_ocr_cost_usd(page_count: int) -> float:
    """USD cost for Mistral OCR on `page_count` pages.

    Args:
        page_count: số trang đã OCR (>= 0)

    Returns:
        Cost USD, rounded to 6 decimals (matches TS implementation)
    """
    if page_count < 0:
        raise ValueError(f"page_count must be >= 0, got {page_count}")
    cost = (page_count / 1000.0) * MISTRAL_OCR_USD_PER_1000_PAGES
    return round(cost, 6)


# ----------------------------------------------------------------------------
# Datalab Marker — ~$4 / 1000 pages (observed from invoices, no use_llm). @phase R221
# use_llm mode costs more; update this constant if that's enabled.
# ----------------------------------------------------------------------------
DATALAB_OCR_USD_PER_1000_PAGES: float = 4.0


def datalab_ocr_cost_usd(page_count: int) -> float:
    """USD cost for Datalab Marker OCR on `page_count` pages."""
    if page_count < 0:
        raise ValueError(f"page_count must be >= 0, got {page_count}")
    cost = (page_count / 1000.0) * DATALAB_OCR_USD_PER_1000_PAGES
    return round(cost, 6)


# ----------------------------------------------------------------------------
# Future (B3/B4 will populate):
#   - voyage_embed_cost_usd(tokens)
#   - anthropic_enrich_cost_usd(input_tokens, output_tokens, model)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Voyage AI — voyage-3-large embedding @ $0.18 / 1M tokens
# Source: https://docs.voyageai.com/docs/pricing (2026)
# ----------------------------------------------------------------------------
VOYAGE_EMBED_USD_PER_M_TOKENS: float = 0.18


def voyage_embed_cost_usd(total_tokens: int) -> float:
    """USD cost for Voyage voyage-3-large embedding on `total_tokens` tokens.

    Args:
        total_tokens: tokens consumed (>= 0)

    Returns:
        Cost USD, rounded to 6 decimals (matches TS voyageEmbedCostUsd).
    """
    if total_tokens < 0:
        raise ValueError(f"total_tokens must be >= 0, got {total_tokens}")
    cost = (total_tokens / 1_000_000.0) * VOYAGE_EMBED_USD_PER_M_TOKENS
    return round(cost, 6)


# ----------------------------------------------------------------------------
# Anthropic Claude Haiku 4.5 — verified anthropic.com pricing (2026-05-30)
# Used for: enrichment step + metadata extraction (cheap tier LLM tasks)
# ----------------------------------------------------------------------------
HAIKU_45_INPUT_USD_PER_M: float = 1.00
HAIKU_45_OUTPUT_USD_PER_M: float = 5.00
HAIKU_45_CACHE_WRITE_1H_USD_PER_M: float = 2.00  # 2.0× base for 1h TTL
HAIKU_45_CACHE_READ_USD_PER_M: float = 0.10      # 0.1× base (90% savings)


def haiku_45_cost_usd(
    input_tokens: int,
    output_tokens: int,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> float:
    """USD cost for one Haiku 4.5 API call with prompt cache support.

    Anthropic Python SDK returns these 4 token fields separately in usage.
    All amounts >= 0; raises ValueError if any is negative.

    Args:
        input_tokens: standard (non-cached) input tokens
        output_tokens: response output tokens
        cache_creation_input_tokens: tokens written to 1h cache (first call)
        cache_read_input_tokens: tokens read from cache hit (subsequent calls)

    Returns:
        Cost USD, rounded to 6 decimals.
    """
    for name, val in [
        ("input_tokens", input_tokens),
        ("output_tokens", output_tokens),
        ("cache_creation_input_tokens", cache_creation_input_tokens),
        ("cache_read_input_tokens", cache_read_input_tokens),
    ]:
        if val < 0:
            raise ValueError(f"{name} must be >= 0, got {val}")

    cost = (
        (input_tokens / 1_000_000.0) * HAIKU_45_INPUT_USD_PER_M
        + (output_tokens / 1_000_000.0) * HAIKU_45_OUTPUT_USD_PER_M
        + (cache_creation_input_tokens / 1_000_000.0) * HAIKU_45_CACHE_WRITE_1H_USD_PER_M
        + (cache_read_input_tokens / 1_000_000.0) * HAIKU_45_CACHE_READ_USD_PER_M
    )
    return round(cost, 6)
