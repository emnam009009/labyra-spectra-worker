"""Extract DOI references from paper full text.

Port labyra-app/src/lib/ai/citations/references-parser.ts.

Strategy:
  1. Find references section header (English + Vietnamese)
  2. Extract DOI-format strings from that section
  3. Capture ±25 chars context per DOI
  4. Fall back to whole-document scan if section header missing

DOI regex matches TS: \\b10\\.\\d{4,9}/[-._;()/:a-zA-Z0-9]*[a-zA-Z0-9] (R168-3.3a)

@phase R167-B5a
"""
from __future__ import annotations

import re

from src.papers.citation_types import ExtractedReference

# DOI scan regex — strict shape mirrors TS (R168-3.3)
_DOI_SCAN_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:a-zA-Z0-9]*[a-zA-Z0-9](?![.\d])")

# Validation regex — must end alphanum (R168-3.3)
_DOI_VALIDATE_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:a-zA-Z0-9]*[a-zA-Z0-9]$")

# Section headers — case insensitive, line-anchored. Match TS regex exactly.
_SECTION_HEADERS = [
    re.compile(r"^[\s\d.]*references[\s.:]*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^[\s\d.]*bibliography[\s.:]*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^[\s\d.]*works cited[\s.:]*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^[\s\d.]*literature cited[\s.:]*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^[\s\d.]*tài liệu tham khảo[\s.:]*$", re.IGNORECASE | re.MULTILINE),
]

# Trailing punctuation to strip from DOI (mirrors TS [.,;)\]\s]+$)
_TRAILING_PUNCT_RE = re.compile(r"[.,;)\]\s]+$")

# Whitespace normalization for context capture
_WHITESPACE_RE = re.compile(r"\s+")

CONTEXT_WINDOW_CHARS = 25
DEFAULT_MAX_RESULTS = 100


def _find_references_section_start(text: str) -> int:
    """Find references section start index. Returns -1 if not found.

    Takes LAST occurrence of header (some papers mention "see references" earlier).
    """
    for pattern in _SECTION_HEADERS:
        matches = list(pattern.finditer(text))
        if matches:
            return matches[-1].start()
    return -1


def extract_dois_from_text(
    full_text: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> list[ExtractedReference]:
    """Extract unique DOIs from paper text.

    Args:
        full_text: OCR markdown/plain text
        max_results: safety cap (default 100; meta-analyses can have 500+)

    Returns:
        Deduplicated list, ordered by appearance.
    """
    if not full_text or not isinstance(full_text, str):
        return []

    section_start = _find_references_section_start(full_text)
    scan_text = full_text[section_start:] if section_start >= 0 else full_text

    results: list[ExtractedReference] = []
    seen: set[str] = set()

    for match in _DOI_SCAN_RE.finditer(scan_text):
        if len(results) >= max_results:
            break

        doi = match.group(0)
        # Strip trailing punctuation/whitespace (regex may be greedy)
        doi = _TRAILING_PUNCT_RE.sub("", doi)

        # Validate strict shape
        if not _DOI_VALIDATE_RE.match(doi):
            continue

        # Dedup case-insensitive (DOIs are case-insensitive per Crossref)
        key = doi.lower()
        if key in seen:
            continue
        seen.add(key)

        # Capture ±25 chars context
        match_start = match.start()
        ctx_start = max(0, match_start - CONTEXT_WINDOW_CHARS)
        ctx_end = min(len(scan_text), match_start + len(doi) + CONTEXT_WINDOW_CHARS)
        context = _WHITESPACE_RE.sub(" ", scan_text[ctx_start:ctx_end]).strip()

        results.append(ExtractedReference(doi=doi, context=context))

    return results
