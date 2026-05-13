"""Anthropic analyzer with type dispatch. @phase R160-spectra-3c-hotfix"""

from __future__ import annotations

import json
import logging
from typing import Any

from anthropic import Anthropic

from src.ai.prompts import build_user_prompt, system_prompt
from src.config import get_settings

logger = logging.getLogger(__name__)


def analyze(parsed: dict[str, Any], metadata: dict[str, Any], locale: str) -> dict[str, Any]:
    settings = get_settings()
    client = Anthropic(api_key=settings.anthropic_api_key)

    spectrum_type = parsed.get("spectrum_type", "xrd")
    system = system_prompt(locale, spectrum_type)
    user = build_user_prompt(parsed, metadata)

    logger.info(
        "AI analyze: type=%s, locale=%s, model=%s, peaks=%d",
        spectrum_type, locale, settings.anthropic_model, len(parsed.get("peaks", [])),
    )

    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=settings.anthropic_max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )

    text_parts = [block.text for block in response.content if block.type == "text"]
    raw = "".join(text_parts).strip()

    if raw.startswith("```"):
        raw = raw.split("```", 2)[-2] if raw.count("```") >= 2 else raw
        raw = raw.removeprefix("json").strip()

    try:
        parsed_json = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.exception("AI returned invalid JSON. Raw: %s", raw[:500])
        raise ValueError(f"AI returned non-JSON: {exc}") from exc

    REQUIRED = {
        "xrd": {"summary", "phases", "overall_confidence"},
        "uvvis": {"summary", "bandgap", "overall_confidence"},
        "uvvis_drs": {"summary", "bandgap", "overall_confidence"},
        "raman": {"summary", "vibrational_modes", "overall_confidence"},
        "ftir": {"summary", "functional_groups", "overall_confidence"},
    }
    required = REQUIRED.get(spectrum_type, {"summary", "overall_confidence"})

    missing = required - set(parsed_json.keys())
    if missing:
        raise ValueError(f"AI response missing fields for {spectrum_type}: {missing}")

    parsed_json["_meta"] = {
        "model": response.model,
        "tokens_in": response.usage.input_tokens,
        "tokens_out": response.usage.output_tokens,
        "locale_used": locale,
        "spectrum_type": spectrum_type,
    }

    return parsed_json
