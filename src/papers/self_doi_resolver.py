"""Self-DOI resolution (Phase 1) — find a paper's OWN DOI + guard title override.

Problem (audit R237): metadata.py extracts the DOI only from page 1 via Gemini,
so when the DOI sits in the footer / page 2-3 (or OCR drops it) the paper gets
no DOI → no Crossref enrichment → "Untitled". Separately, R228 overwrites the
OCR title with the Crossref title *unconditionally*, so a hallucinated/misread
DOI would silently replace the title with a DIFFERENT paper's.

Phase 1 (deterministic, no LLM — LLM DOI reading is the very source of the
hallucination risk):

  1. extract_self_doi(pages) — scan the FIRST 3 pages for a LABELLED DOI
     (`doi:` / `doi.org/`). A labelled DOI on the opening pages is a strong
     self-DOI signal (web/library guidance: the DOI is printed on the first
     page near the title or in the header/footer). We deliberately do NOT grab
     unlabelled DOI-shaped strings here — those are usually references — to keep
     Trust > Coverage. Text after a "References"/"Bibliography" header is cut off.

  2. should_override_title(...) — multi-tier guard before letting a
     resolved-by-DOI title overwrite the OCR title (gap A). Returns (override?,
     reason). Tiers, in order:
       a. OCR title empty / "Untitled"      → override (nothing to lose)
       b. ≥1 author family-name in common   → override (same paper even if the
                                               OCR title is badly misread)
       c. token-set Jaccard(title) ≥ 0.35   → override (e.g. "Please" vs "Phage"
                                               ≈ 0.78 passes; a wholly different
                                               paper ≈ 0.1 is rejected)
       d. otherwise                         → DO NOT override (keep OCR title,
                                               caller flags doiTitleMismatch)

Reuses jaccard_similarity() + the DOI regex/section finder already in the repo
(no new dependency).

@phase R237bm (doi-resolution Phase 1)
"""
from __future__ import annotations

import re

from pydantic import BaseModel

# Reuse existing helpers (same package) — keep one source of truth.
from src.papers.google_books import jaccard_similarity
from src.papers.references_parser import (
    _DOI_VALIDATE_RE,
    _find_references_section_start,
)

# Guard A: minimum token-set Jaccard to accept that a resolved title is the
# SAME paper as the OCR title. Low on purpose — this guard only rejects a DOI
# that resolved to a wholly different paper; it must NOT block fixing OCR typos
# (e.g. "Please-Inspired ..." → "Phage-Inspired ..." ≈ 0.78). See doi-resolution.md.
TITLE_OVERRIDE_MIN_JACCARD = 0.35

# Number of leading pages to scan for the self-DOI.
SELF_DOI_PAGE_WINDOW = 3

# Labelled DOI: "doi:10.x/..." or "(https://)(dx.)doi.org/10.x/...". Two capture
# groups (label form vs URL form); whichever matched is the DOI.
_LABELLED_DOI_RE = re.compile(
    r"(?:https?://)?(?:dx\.)?doi\.org/(10\.\d{4,9}/[-._;()/:a-z0-9]+)"
    r"|doi[:\s]+(10\.\d{4,9}/[-._;()/:a-z0-9]+)",
    re.IGNORECASE,
)

_TRAILING_PUNCT_RE = re.compile(r"[.,;)\]\s]+$")


class SelfDoiResult(BaseModel):
    """Outcome of self-DOI extraction. found=False = leave DOI to other steps."""

    found: bool = False
    doi: str = ""
    source: str = ""  # 'page-text' when found here


def _normalize_doi(doi: str) -> str:
    return _TRAILING_PUNCT_RE.sub("", doi.strip())


def extract_self_doi(pages_text: list[str]) -> SelfDoiResult:
    """Find a labelled self-DOI in the first few pages. Best-effort, never raises.

    Args:
        pages_text: OCR markdown per page, in order (page 1 first).

    Returns:
        SelfDoiResult(found=True, doi, source='page-text') or found=False.
    """
    if not pages_text:
        return SelfDoiResult()

    head = "\n".join(pages_text[:SELF_DOI_PAGE_WINDOW])
    # Cut at a references/bibliography header so we never pick a reference's DOI.
    ref_start = _find_references_section_start(head)
    scan = head[:ref_start] if ref_start >= 0 else head

    match = _LABELLED_DOI_RE.search(scan)
    if not match:
        return SelfDoiResult()

    doi = _normalize_doi(match.group(1) or match.group(2) or "")
    if doi and _DOI_VALIDATE_RE.match(doi):
        return SelfDoiResult(found=True, doi=doi, source="page-text")
    return SelfDoiResult()


def choose_self_doi(gemini_doi: str, pages_text: list[str]) -> tuple[str, str]:
    """Pick the authoritative self-DOI for a paper.

    The deterministic labelled DOI printed on the opening pages (a "doi.org/10..."
    URL or a "DOI: 10..." line) is GROUND TRUTH. The metadata LLM can silently
    truncate it (e.g. "10.1002/advs.202105135" -> "10.1002/adv.202105135"), which
    then fails to resolve and leaves doiVerified=False (the amber-triangle bug).

    extract_self_doi() already cuts at the references section and format-validates,
    so it returns the paper's OWN DOI safely — never a reference's. We therefore
    PREFER it whenever found, and only fall back to the LLM value when no labelled
    DOI is present (e.g. noisy OCR where the URL didn't survive). A title-based
    reverse lookup remains the caller's final fallback when neither yields a DOI.

    Returns (doi, source). source ∈ {"page-text", "gemini", ""}.

    @phase R238-doi-deterministic-first
    """
    recovered = extract_self_doi(pages_text)
    if recovered.found:
        return recovered.doi, recovered.source  # "page-text"
    gemini = (gemini_doi or "").strip()
    if gemini:
        return gemini, "gemini"
    return "", ""


_NAME_PARTICLES = {
    "van", "von", "der", "den", "del", "dela", "los", "las", "san", "bin", "ibn",
}


def _name_tokens(authors: list[str]) -> set[str]:
    """Lower-cased name tokens (≥3 letters, particles dropped) from any format.

    Order-agnostic on purpose: "Smith, John" (Crossref) and "John Smith" (Gemini)
    both yield {smith, john}; Vietnamese "Nguyen Van A" yields {nguyen} ("van" is
    a particle, "a" is a 1-char initial). Initials and 1-2 char tokens are
    dropped so they never produce spurious matches.
    """
    out: set[str] = set()
    for a in authors or []:
        if not isinstance(a, str):
            continue
        for tok in re.findall(r"[a-z]{3,}", a.lower()):
            if tok not in _NAME_PARTICLES:
                out.add(tok)
    return out


def authors_overlap(a: list[str], b: list[str]) -> bool:
    """True if the two author lists share at least one name token (≥3 chars).

    Intentionally lenient — it only ever GRANTS a title override (tier b), and is
    reached only when the DOI is the paper's own and the titles already disagree.
    """
    return bool(_name_tokens(a) & _name_tokens(b))


def should_override_title(
    ocr_title: str,
    ocr_authors: list[str],
    resolved_title: str,
    resolved_authors: list[str],
) -> tuple[bool, str]:
    """Guard A: decide whether a resolved-by-DOI title may replace the OCR title.

    Returns (override, reason). reason is a short machine tag for logging.
    """
    resolved = (resolved_title or "").strip()
    if not resolved:
        return (False, "no_resolved_title")

    ocr = (ocr_title or "").strip()
    # Tier a: nothing to lose.
    if not ocr or ocr.lower() == "untitled":
        return (True, "ocr_empty")

    # Tier b: author family-name overlap → same paper even if title misread.
    if authors_overlap(ocr_authors, resolved_authors):
        return (True, "author_overlap")

    # Tier c: titles are close enough (token-set Jaccard).
    if jaccard_similarity(ocr, resolved) >= TITLE_OVERRIDE_MIN_JACCARD:
        return (True, "title_jaccard")

    # Tier d: looks like a different paper — keep OCR title, caller flags it.
    return (False, "title_mismatch")
