"""Reference-list extraction — DOI-anchored (R237bo, branch B Phase 1, revised).

Goal (user req #3/#4, revised): list the paper's references that carry a DOI,
numbered in order of appearance, each with the raw reference line — so the UI can
show a proper numbered reference list (not just bare DOIs).

WHY DOI-ANCHORED (decision A, after measuring on real papers):
The first attempt split the references section by line-start markers ([1]/1./(1))
plus a paragraph fallback. On real papers it failed two ways:
  - Many papers (reviews, Elsevier/Frontiers author-year style) have NO numbered
    markers and the OCR often drops the literal "References" header, so the
    section boundary is lost.
  - With no boundary, marker/paragraph splitting scanned the whole body and
    turned headings ("# 2.4 …"), tables ("Table 3 …") and figures into fake
    "references" — observed 191 entries for one paper, ~half rubbish.

A reference without a DOI cannot be told apart from body text deterministically
once the header is gone, and guessing produces rubbish. So we anchor on the DOI:
a DOI is hard evidence that the surrounding line is a citation. This:
  - kills the heading/table rubbish (no DOI → not captured),
  - never does worse than the old extract_dois_from_text (same DOIs found; we
    just add a number + the raw reference line),
  - is deterministic, $0, reuses the repo's DOI regex + section finder.

Trade-off (accepted): references with NO DOI are not listed. Most materials-
science references carry a DOI; this is the safe failure (missing, never wrong).
Enriching DOI-less refs later (GROBID/Anystyle) would not need re-OCR.

@phase R237bo  (supersedes the marker-split R237bn)
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

DEFAULT_MAX_REFERENCES = 300  # safety cap (meta-analyses can be long)
RAW_TEXT_CAP = 600  # a single reference line is short; cap stored text


class ExtractedReferenceEntry(BaseModel):
    """One DOI-bearing reference: position, raw line, DOI."""

    model_config = ConfigDict(extra="forbid")

    number: int = Field(ge=1)
    raw_text: str
    doi: str  # always present (entries are anchored on a DOI)


def _clean(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def extract_references(
    full_text: str,
    max_results: int = DEFAULT_MAX_REFERENCES,
) -> list[ExtractedReferenceEntry]:
    """Extract DOI-bearing references in order, with the raw reference line.

    Scans the references section when its header is present (more precise),
    otherwise the whole text — safe either way because only DOI-bearing lines
    are captured. Deduplicates by DOI; numbers by order of appearance.

    Best-effort, never raises.
    """
    if not full_text or not isinstance(full_text, str):
        return []

    start = _find_references_section_start(full_text)
    scan = full_text[start:] if start >= 0 else full_text

    entries: list[ExtractedReferenceEntry] = []
    seen: set[str] = set()

    for m in _DOI_SCAN_RE.finditer(scan):
        if len(entries) >= max_results:
            break
        doi = _TRAILING_PUNCT_RE.sub("", m.group(0))
        if not _DOI_VALIDATE_RE.match(doi):
            continue
        key = doi.lower()
        if key in seen:
            continue
        seen.add(key)

        # raw_text = the line (between newlines) that contains the DOI.
        line_start = scan.rfind("\n", 0, m.start()) + 1
        line_end = scan.find("\n", m.end())
        if line_end == -1:
            line_end = len(scan)
        raw = _clean(scan[line_start:line_end])

        entries.append(
            ExtractedReferenceEntry(
                number=len(entries) + 1,
                raw_text=raw[:RAW_TEXT_CAP],
                doi=doi,
            )
        )

    return entries
