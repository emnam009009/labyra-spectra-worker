"""Reference-list extraction (R237bn, branch B Phase 1).

Goal (user requirements #3/#4): list ALL references in a paper, numbered in the
paper's own order, with the DOI when present — not just the DOI-bearing ones.

The existing extract_dois_from_text() only captures DOI-format strings (dropping
any reference without a DOI) and has no ordering. This module parses the
references section into ordered entries, each keeping the raw reference text so
nothing is lost; the DOI (if any) is pulled with the repo's existing regex.

Design (Trust > Coverage, deterministic — no LLM):
  - We do NOT parse author/title/year fields here. Rule/regex field-parsing has
    low recall (literature: F1 ≈ 0.33 vs GROBID 0.89); doing it badly would
    invent data. For DOI-bearing refs the authoritative author/title/year comes
    from Crossref/OpenAlex (lookup_doi). For DOI-less refs we keep the raw text
    so the reference is still listed + numbered, and can be enriched later
    (GROBID/Anystyle) without re-OCR.
  - Primary strategy: split on line-start numeric markers ([1] / (1) / 1.).
  - Fallback (un-numbered / author-year lists): split on blank lines, number by
    order of appearance.

Reuses _find_references_section_start, _DOI_SCAN_RE, _DOI_VALIDATE_RE,
_TRAILING_PUNCT_RE, _WHITESPACE_RE from references_parser.py (one source of truth).

@phase R237bn
"""
from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from src.papers.references_parser import (
    _DOI_SCAN_RE,
    _DOI_VALIDATE_RE,
    _TRAILING_PUNCT_RE,
    _WHITESPACE_RE,
    _find_references_section_start,
)

DEFAULT_MAX_REFERENCES = 300  # meta-analyses can be long; safety cap
RAW_TEXT_CAP = 600  # a single reference is short; cap stored text
MIN_REF_LEN = 20  # fallback: ignore fragments shorter than this
MIN_NUMBERED_MARKERS = 3  # need at least this many to trust numbered-split

# Line-start reference markers: "[12]", "(12)", or "12." followed by whitespace.
# MULTILINE so only line beginnings match (mid-line "[12]" citations ignored).
_ENTRY_MARKER_RE = re.compile(
    r"^[ \t]*(?:\[(\d{1,4})\]|\((\d{1,4})\)|(\d{1,4})\.)[ \t]+",
    re.MULTILINE,
)


class ExtractedReferenceEntry(BaseModel):
    """One reference from the paper's reference list (numbered, raw, optional DOI)."""

    model_config = ConfigDict(extra="forbid")

    number: int = Field(ge=1)
    """Position in the paper's reference list (1-based)."""

    raw_text: str
    """The reference string as printed (whitespace-normalised, capped)."""

    doi: str | None = None
    """DOI if one appears in this entry, else None."""


def _clean(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _doi_in(text: str) -> str | None:
    m = _DOI_SCAN_RE.search(text)
    if not m:
        return None
    doi = _TRAILING_PUNCT_RE.sub("", m.group(0))
    return doi if _DOI_VALIDATE_RE.match(doi) else None


def _make_entry(number: int, raw: str) -> ExtractedReferenceEntry:
    return ExtractedReferenceEntry(
        number=number,
        raw_text=raw[:RAW_TEXT_CAP],
        doi=_doi_in(raw),
    )


def extract_references(
    full_text: str,
    max_results: int = DEFAULT_MAX_REFERENCES,
) -> list[ExtractedReferenceEntry]:
    """Parse the references section into ordered entries. Best-effort, never raises.

    Args:
        full_text: concatenated OCR text from all pages.
        max_results: safety cap on number of entries.

    Returns:
        Ordered list of ExtractedReferenceEntry (may be empty).
    """
    if not full_text or not isinstance(full_text, str):
        return []

    start = _find_references_section_start(full_text)
    section = full_text[start:] if start >= 0 else full_text

    markers = list(_ENTRY_MARKER_RE.finditer(section))
    entries: list[ExtractedReferenceEntry] = []

    if len(markers) >= MIN_NUMBERED_MARKERS:
        # Numbered-split: each marker begins an entry that runs to the next.
        for i, m in enumerate(markers):
            if len(entries) >= max_results:
                break
            number = int(m.group(1) or m.group(2) or m.group(3))
            body_start = m.end()
            body_end = markers[i + 1].start() if i + 1 < len(markers) else len(section)
            raw = _clean(section[body_start:body_end])
            if raw:
                entries.append(_make_entry(number, raw))
        return entries

    # Fallback: blank-line paragraphs, numbered by appearance.
    n = 0
    for chunk in re.split(r"\n\s*\n", section):
        raw = _clean(chunk)
        if len(raw) < MIN_REF_LEN:
            continue  # skips the bare "References" header + fragments
        n += 1
        if n > max_results:
            break
        entries.append(_make_entry(n, raw))
    return entries
