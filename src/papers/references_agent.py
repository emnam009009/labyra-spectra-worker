"""Agent-assisted reference extraction — source C (R237bu).

Last-resort fallback when both Crossref (A) and PDF DOI-anchored (B) yield
nothing: e.g. a paper with no resolvable self-DOI whose references section is
author-year style with no inline DOIs. A low-temperature LLM (Gemini Flash, T2)
*structures* the OCR'd references section into entries.

CRITICAL — this is grounded EXTRACTION, not generation:
  - The model only structures text that is already present; it must not invent
    references, DOIs, authors, or years.
  - DOIs are NEVER trusted from the model. After extraction, every returned DOI
    is verified to appear VERBATIM in the source text (deterministic check); if
    not, it is dropped. This is the hard guard against hallucinated DOIs.
  - All agent entries are stored with confidence='unverified'.

Only invoked when A and B are empty, so cost is incurred rarely.

@phase R237bu
"""
from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field

from src.config import get_settings
from src.papers._gemini_client import extract_json
from src.papers.references_extractor import (
    RAW_TEXT_CAP,
    ExtractedReferenceEntry,
    _clean,
)
from src.papers.references_parser import (
    _DOI_SCAN_RE,
    _DOI_VALIDATE_RE,
    _TRAILING_PUNCT_RE,
    _find_references_section_start,
)

logger = logging.getLogger(__name__)

MAX_AGENT_REFERENCES = 80  # cap LLM output size / cost
_SECTION_INPUT_CAP = 24000  # chars of references section fed to the model

_SYSTEM_INSTRUCTION = (
    "You extract the reference/bibliography entries from the provided text of a "
    "scientific paper's references section. Return each reference as a structured "
    "entry. STRICT RULES: (1) Only use text that is actually present — never "
    "invent, complete, or guess any reference, DOI, author, title, or year. "
    "(2) Copy each reference's full text verbatim into raw_text. (3) Include a doi "
    "ONLY if a DOI string literally appears in that reference's text; otherwise "
    "leave doi empty. Do not construct or normalize DOIs. (4) Fill title/authors/"
    "year only when clearly present; otherwise leave them empty. (5) Preserve the "
    "order of appearance. Do not include section headings, figures, tables, or "
    "body text — only bibliography entries."
)


class _AgentReference(BaseModel):
    model_config = ConfigDict(extra="ignore")
    raw_text: str = Field(default="")
    title: str | None = None
    authors: list[str] | None = None
    year: int | None = None
    doi: str | None = None


class _AgentReferenceList(BaseModel):
    model_config = ConfigDict(extra="ignore")
    references: list[_AgentReference] = Field(default_factory=list)


def _verified_doi(candidate: str | None, source_text_lower: str) -> str | None:
    """Return the DOI only if it (a) is well-formed AND (b) appears verbatim in
    the source text. Otherwise None — the model is never trusted for DOIs."""
    if not candidate or not isinstance(candidate, str):
        return None
    doi = _TRAILING_PUNCT_RE.sub("", candidate.strip()).lower()
    doi = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/").removeprefix("doi:")
    if not _DOI_VALIDATE_RE.match(doi):
        return None
    return doi if doi in source_text_lower else None


def extract_references_with_agent(
    full_text: str,
) -> tuple[list[ExtractedReferenceEntry], int, int]:
    """Structure the references section via the LLM (source C).

    Returns (entries, input_tokens, output_tokens). entries is empty on any
    failure (caller already exhausted A and B). Never raises.
    """
    if not full_text or not isinstance(full_text, str):
        return [], 0, 0

    start = _find_references_section_start(full_text)
    section = (full_text[start:] if start >= 0 else full_text)[:_SECTION_INPUT_CAP]
    section_lower = section.lower()

    settings = get_settings()
    try:
        parsed, in_tok, out_tok = extract_json(
            model=settings.gemini_model_metadata,
            prompt=section,
            schema=_AgentReferenceList,
            system_instruction=_SYSTEM_INSTRUCTION,
            max_tokens=6000,
            temperature=0.0,
            thinking_budget=0,
        )
    except Exception as exc:  # noqa: BLE001 — defensive; agent is best-effort
        logger.warning("references_agent_failed err=%s", exc)
        return [], 0, 0

    if parsed is None:
        return [], 0, 0

    entries: list[ExtractedReferenceEntry] = []
    for r in parsed.references:
        if len(entries) >= MAX_AGENT_REFERENCES:
            break
        raw = _clean(r.raw_text or "")
        if len(raw) < 12:
            continue
        # DOI: trust the verbatim text, not the model. Prefer a DOI actually
        # found in the raw line; fall back to the model's doi only if it too is
        # present verbatim in the section.
        doi = None
        m = _DOI_SCAN_RE.search(raw)
        if m:
            cand = _TRAILING_PUNCT_RE.sub("", m.group(0))
            if _DOI_VALIDATE_RE.match(cand):
                doi = cand.lower()
        if doi is None:
            doi = _verified_doi(r.doi, section_lower)
        entries.append(
            ExtractedReferenceEntry(
                number=len(entries) + 1,
                raw_text=raw[:RAW_TEXT_CAP],
                doi=doi,
            )
        )
    logger.info("references_agent_done entries=%d in_tok=%d out_tok=%d", len(entries), in_tok, out_tok)
    return entries, in_tok, out_tok
