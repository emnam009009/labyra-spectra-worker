"""Google Gemini client singleton for paper pipeline LLM calls.

Used by metadata.py (extract) + enrich.py (chunk context). NOT used by
src/ai/analyzer.py — that stays on Anthropic for scientific output stability.

Uses google-genai SDK 2.3+ (Pydantic-native JSON schema support).

R177-1a-hotfix: thinking_budget=0 default — Gemini 3 series charges thoughts
tokens at output rate ($3/1M). For structured metadata + short enrichment
tasks, reasoning offers no quality gain but doubles cost. Caller can override
via thinking_budget param if reasoning needed.

@phase R177-1a → R177-1a-hotfix
"""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from typing import TypeVar

from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel

from src.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Strip ```json fences if model adds them despite response_mime_type=application/json
_MARKDOWN_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)


@lru_cache(maxsize=1)
def get_gemini_client() -> genai.Client:
    """Singleton Gemini client (lru_cache for worker instance reuse)."""
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not configured. Set in .env.local for dev or mount "
            "via Cloud Run secret in production."
        )
    return genai.Client(api_key=settings.gemini_api_key)


def _build_config(
    *,
    system_instruction: str | None,
    max_tokens: int,
    temperature: float,
    thinking_budget: int,
    response_mime_type: str | None = None,
    response_json_schema: dict | None = None,
) -> genai_types.GenerateContentConfig:
    """Build GenerateContentConfig with consistent defaults."""
    kwargs: dict = {
        "max_output_tokens": max_tokens,
        "temperature": temperature,
        "thinking_config": genai_types.ThinkingConfig(thinking_budget=thinking_budget),
    }
    if system_instruction:
        kwargs["system_instruction"] = system_instruction
    if response_mime_type:
        kwargs["response_mime_type"] = response_mime_type
    if response_json_schema:
        kwargs["response_json_schema"] = response_json_schema
    return genai_types.GenerateContentConfig(**kwargs)


def _count_output_tokens(meta) -> int:
    """Sum candidates + thoughts. Gemini 3 charges both at output rate."""
    if meta is None:
        return 0
    candidates = meta.candidates_token_count or 0
    thoughts = meta.thoughts_token_count or 0
    return candidates + thoughts


def extract_text(
    model: str,
    prompt: str,
    *,
    system_instruction: str | None = None,
    max_tokens: int = 500,
    temperature: float = 0.0,
    thinking_budget: int = 0,
) -> tuple[str, int, int]:
    """Plain text completion (for enrichment task).

    Args:
        thinking_budget: tokens model can spend on reasoning. 0=disabled
            (default, recommended for structured tasks). Pass -1 for unlimited.

    Returns:
        (text, input_tokens, output_tokens). On any error, returns ("", 0, 0).
        output_tokens includes thoughts when thinking_budget > 0.

    Caller is responsible for parsing / fallback handling. Never raises.
    """
    try:
        client = get_gemini_client()
        config = _build_config(
            system_instruction=system_instruction,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_budget=thinking_budget,
        )
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        text = response.text or ""
        meta = response.usage_metadata
        in_tok = (meta.prompt_token_count or 0) if meta else 0
        out_tok = _count_output_tokens(meta)
        return text, in_tok, out_tok
    except Exception as exc:  # noqa: BLE001 — best-effort
        logger.warning("gemini_extract_text_failed model=%s err=%s", model, exc)
        return "", 0, 0


def extract_json(
    model: str,
    prompt: str,
    schema: type[T],
    *,
    system_instruction: str | None = None,
    max_tokens: int = 500,
    temperature: float = 0.0,
    thinking_budget: int = 0,
) -> tuple[T | None, int, int]:
    """Structured JSON output via Pydantic schema (for metadata extract).

    Uses response_mime_type=application/json + response_json_schema so Gemini
    constrains output to match schema. Parses to Pydantic model.

    Returns:
        (parsed | None, input_tokens, output_tokens). On any failure, parsed
        is None — caller falls back to defaults.
    """
    raw = ""
    try:
        client = get_gemini_client()
        config = _build_config(
            system_instruction=system_instruction,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_budget=thinking_budget,
            response_mime_type="application/json",
            response_json_schema=schema.model_json_schema(),
        )
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        raw = response.text or ""
        meta = response.usage_metadata
        in_tok = (meta.prompt_token_count or 0) if meta else 0
        out_tok = _count_output_tokens(meta)

        cleaned = _MARKDOWN_FENCE_RE.sub("", raw).strip()
        parsed = json.loads(cleaned)
        return schema.model_validate(parsed), in_tok, out_tok
    except json.JSONDecodeError as exc:
        logger.warning("gemini_json_parse_failed model=%s err=%s raw=%s", model, exc, raw[:200])
        return None, 0, 0
    except Exception as exc:  # noqa: BLE001 — best-effort
        logger.warning("gemini_extract_json_failed model=%s err=%s", model, exc)
        return None, 0, 0
