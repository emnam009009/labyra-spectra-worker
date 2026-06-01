"""Pre-translate Lớp 1 — translate high-value sections at upload time.

Researchers read abstract -> conclusion -> headings first (skim-to-deep), so we
translate exactly those into the tenant's default language; they're then instant
when the paper opens. The body is left to on-demand translation.

Eligibility (should_pretranslate): article-type + size-bounded only. Skipped when
the paper is already in the target language (en->en).

Storage: tenants/{tid}/papers/{paperId}/pretranslations/{lang} (one doc).

Best-effort + non-fatal: orchestrator wraps the step; any failure is logged and
the pipeline continues.

@phase R223
"""
from __future__ import annotations

import logging
from typing import NamedTuple

from google.cloud import firestore  # type: ignore[import-untyped]

from src.config import get_settings
from src.papers._gemini_client import extract_text
from src.papers.chunking import _detect_section, _split_paragraphs
from src.papers.types import OcrResult

logger = logging.getLogger(__name__)

# Strategy §2.4 — pattern + cost don't fit these document types.
_EXCLUDED_TYPES = {"book", "thesis", "dissertation", "monograph", "presentation"}
_MAX_PAGES = 50
_MAX_CHARS = 200_000

_ABSTRACT_NAMES = ("abstract", "summary")
_CONCLUSION_NAMES = (
    "conclusion",
    "conclusions",
    "concluding",
    "conclusions and",
    "summary and conclusion",
    "outlook",
)

# Gemini pricing (gemini-3-flash-preview): $0.50/1M in, $3.00/1M out.
_GEMINI_IN_USD_PER_M = 0.50
_GEMINI_OUT_USD_PER_M = 3.00

_LANG_NAME = {
    "en": "English",
    "vi": "Vietnamese",
    "zh": "Chinese (Simplified)",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
}


class PretranslateSections(NamedTuple):
    """High-value sections pulled from a paper's OCR for pre-translation."""

    abstract: str
    conclusion: str
    headings: list[str]


def should_pretranslate(document_type: str, num_pages: int, total_chars: int) -> bool:
    """Article-type + size-bounded only (strategy §2.4)."""
    if (document_type or "").strip().lower() in _EXCLUDED_TYPES:
        return False
    if num_pages > _MAX_PAGES:
        return False
    return total_chars <= _MAX_CHARS


def _find_section(sections: dict[str, list[str]], names: tuple[str, ...]) -> str:
    """Joined text of the first heading whose name matches `names` (case-insensitive)."""
    for heading, parts in sections.items():
        h = heading.strip().lower()
        if any(h == n or h.startswith(n) for n in names):
            return "\n\n".join(parts).strip()
    return ""


def extract_sections(ocr_result: OcrResult) -> PretranslateSections:
    """Pull abstract, conclusion, and the ordered list of section headings from OCR.

    Reuses chunking's heading/paragraph detection so section boundaries match the
    chunker exactly.
    """
    sections: dict[str, list[str]] = {}
    headings: list[str] = []
    current = ""
    for page in ocr_result.pages:
        for para in _split_paragraphs(page.text):
            lines = para.split("\n", 1)
            heading = _detect_section(lines[0])
            if heading:
                current = heading
                if heading not in headings:
                    headings.append(heading)
                body = lines[1].strip() if len(lines) > 1 else ""
                if body:
                    sections.setdefault(current, []).append(body)
            elif current:
                sections.setdefault(current, []).append(para)
    return PretranslateSections(
        abstract=_find_section(sections, _ABSTRACT_NAMES),
        conclusion=_find_section(sections, _CONCLUSION_NAMES),
        headings=headings,
    )


def _tenant_default_language(db: firestore.Client, tenant_id: str) -> str:
    """Tenant default target language from aiContext/main (default 'en')."""
    try:
        snap = db.document(f"tenants/{tenant_id}/aiContext/main").get()
        if snap.exists:
            lang = (snap.to_dict() or {}).get("defaultLanguage")
            if isinstance(lang, str) and lang.strip():
                return lang.strip().lower()
    except Exception as exc:
        logger.warning("pretranslate_lang_read_failed tenant=%s err=%s", tenant_id, exc)
    return "en"


def _translate(text: str, target_name: str, max_tokens: int) -> tuple[str, int, int]:
    """Translate `text` into `target_name`. Returns (translation, in_tok, out_tok)."""
    if not text.strip():
        return "", 0, 0
    settings = get_settings()
    system = (
        f"Translate scientific text into {target_name} for an expert reader. "
        "Keep chemical formulae, units, acronyms, citation markers, and equations "
        "verbatim (do not transliterate). Output ONLY the translation, no preamble."
    )
    return extract_text(
        model=settings.gemini_model_pretranslate,
        prompt=text,
        system_instruction=system,
        max_tokens=max_tokens,
        temperature=0.2,
    )


def run_pretranslate_step(
    db: firestore.Client,
    tenant_id: str,
    paper_id: str,
    ocr_result: OcrResult,
    document_type: str,
    source_language: str,
    abstract: str = "",
) -> None:
    """Pre-translate abstract/conclusion/headings into the tenant default language.

    Best-effort + non-fatal. Skipped when the paper is ineligible or already in
    the target language (en->en).
    """
    if not should_pretranslate(document_type, ocr_result.page_count, len(ocr_result.full_text)):
        logger.info("pretranslate_skip_ineligible paper=%s type=%s", paper_id, document_type)
        return

    target = _tenant_default_language(db, tenant_id)
    if (source_language or "").strip().lower() == target:
        logger.info("pretranslate_skip_same_language paper=%s lang=%s", paper_id, target)
        return

    target_name = _LANG_NAME.get(target, "English")
    sections = extract_sections(ocr_result)
    # Prefer the metadata-extracted abstract (reliable even when there's no
    # "Abstract" heading, e.g. review articles); fall back to section extraction.
    abstract_src = (abstract or "").strip() or sections.abstract
    if not abstract_src and not sections.conclusion and not sections.headings:
        logger.info("pretranslate_skip_no_sections paper=%s", paper_id)
        return

    budget = get_settings().gemini_max_tokens_pretranslate
    in_tok_total = 0
    out_tok_total = 0

    abstract_t, i_ab, o_ab = _translate(abstract_src, target_name, budget)
    conclusion_t, i_co, o_co = _translate(sections.conclusion, target_name, budget)
    in_tok_total += i_ab + i_co
    out_tok_total += o_ab + o_co

    # Headings: translate as one newline-joined block, then split back by line.
    # Only keep the result if the line count matches (otherwise alignment is lost).
    headings_map: dict[str, str] = {}
    if sections.headings:
        joined = "\n".join(sections.headings)
        translated, i_he, o_he = _translate(joined, target_name, budget)
        in_tok_total += i_he
        out_tok_total += o_he
        out_lines = [ln.strip() for ln in translated.split("\n") if ln.strip()]
        if len(out_lines) == len(sections.headings):
            headings_map = dict(zip(sections.headings, out_lines, strict=False))

    cost_usd = (
        in_tok_total * _GEMINI_IN_USD_PER_M + out_tok_total * _GEMINI_OUT_USD_PER_M
    ) / 1_000_000
    logger.info(
        "pretranslate_cost paper=%s lang=%s in_tok=%d out_tok=%d cost_usd=%.6f",
        paper_id, target, in_tok_total, out_tok_total, cost_usd,
    )

    try:
        db.document(f"tenants/{tenant_id}/papers/{paper_id}/pretranslations/{target}").set({
            "abstract": abstract_t,
            "conclusion": conclusion_t,
            "headings": headings_map,
            "sourceLanguage": (source_language or "").strip().lower(),
            "targetLanguage": target,
            "translatedAt": firestore.SERVER_TIMESTAMP,
        })
        logger.info("pretranslate_done paper=%s lang=%s", paper_id, target)
    except Exception as exc:
        logger.warning("pretranslate_write_failed paper=%s err=%s", paper_id, exc)
