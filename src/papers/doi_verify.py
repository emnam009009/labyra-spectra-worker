"""R242: verify-then-reverse-lookup for a paper's own (self) DOI.

A DOI printed in a PDF — or read by the metadata LLM — can be WRONG: an OCR
misread, a truncation, or the "Cite this" / footer line of a DIFFERENT paper
that happens to sit on the opening page. Resolving such a DOI is NOT enough,
because a wrong-but-valid DOI still resolves (to the wrong record). The resolved
record must actually be THIS paper.

Two-candidate reconciliation, Crossref/OpenAlex as the authoritative arbiter:

  1. VERIFY the extracted candidate — resolve it, then compare the publisher's
     canonical title to our title (order-independent token-set Jaccard). A strong
     match means the candidate points at this paper -> trust it. (Cheap: when the
     printed DOI is right, this is the only network call and we stop here.)

  2. If the candidate does not resolve, or resolves to a DIFFERENT paper, or is
     missing entirely -> REVERSE-LOOKUP the DOI by bibliographic metadata at
     Crossref (`query.bibliographic`, which already enforces Jaccard >= 0.7
     internally). This is candidate #2 and it is title-agnostic to how the DOI
     was (mis)printed, so it recovers the right DOI for the wrong-DOI case.

  3. Neither path yields a title-matched DOI -> keep the candidate as a
     best-effort hint but report it as UNVERIFIED, so the caller can surface the
     amber "unverified" state instead of attaching a confidently-wrong DOI.

Design notes:
  * A wrong DOI is worse than no DOI: it poisons the topic classifier and the
    citation graph. Hence every ACCEPTED DOI is title-checked against the
    registry, and the bar is deliberately strict.
  * Journal-agnostic: no per-publisher prefix heuristics (10.1021 etc.), which
    cannot disambiguate two equally-valid DOIs from the same publisher and rot
    over time. Crossref already owns that knowledge.

@phase R242
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Minimum token-set Jaccard to ACCEPT an extracted candidate as pointing to this
# paper. Set just below the reverse-lookup bar (0.7): the candidate was
# physically printed/extracted, and a false-negative here merely defers to
# reverse-lookup — which re-finds the same correct DOI from the title. A
# false-POSITIVE (a wrong DOI accepted as verified) is the real harm, so this
# stays high enough to reject a mismatched record.
VERIFY_TITLE_THRESHOLD = 0.6

# Below this, a title is too short to disambiguate a DOI reliably.
_MIN_TITLE_LEN = 10


class DoiVerifyResult(BaseModel):
    """Outcome of self-DOI verification.

    `source` is one of:
      - "verified"        the extracted candidate resolved AND its title matched
      - "reverse-lookup"  recovered by Crossref bibliographic search on the title
      - "unverified"      a candidate exists but could not be confirmed
      - "empty"           no candidate and nothing recovered
    """

    doi: str = ""
    source: str = ""
    resolved_title: str = ""
    title_match: float = 0.0

    @property
    def trusted(self) -> bool:
        """True only when the DOI is confirmed to be THIS paper's."""
        return self.source in ("verified", "reverse-lookup")


def verify_self_doi(
    candidate_doi: str | None,
    title: str | None,
    authors: list[str] | None = None,
    year: int | None = None,
) -> DoiVerifyResult:
    """Verify an extracted self-DOI, reverse-looking-up by title if it fails.

    Never guesses: returns an unverified/empty result rather than attaching a
    DOI whose resolved title does not match. See module docstring.
    """
    # Imported lazily to mirror the orchestrator's call-site pattern and avoid a
    # heavier import graph at module load.
    from src.papers.crossref import reverse_lookup_doi
    from src.papers.google_books import jaccard_similarity
    from src.papers.journal_resolve import resolve_journal_from_doi

    cand = (candidate_doi or "").strip()
    ttl = (title or "").strip()

    # 1. Verify the extracted candidate actually points to THIS paper.
    if cand and len(ttl) >= _MIN_TITLE_LEN:
        jr = resolve_journal_from_doi(cand)
        if jr.doi_found and jr.title:
            score = jaccard_similarity(ttl, jr.title)
            if score >= VERIFY_TITLE_THRESHOLD:
                logger.info("doi_verified doi=%s score=%.2f", cand, score)
                return DoiVerifyResult(
                    doi=cand,
                    source="verified",
                    resolved_title=jr.title,
                    title_match=score,
                )
            logger.info(
                "doi_candidate_title_mismatch doi=%s score=%.2f resolved=%r ours=%r",
                cand,
                score,
                jr.title[:60],
                ttl[:60],
            )
        else:
            logger.info("doi_candidate_unresolved doi=%s", cand)

    # 2. Reverse-lookup by title — handles a wrong OR a missing candidate.
    if len(ttl) >= _MIN_TITLE_LEN:
        rev = reverse_lookup_doi(ttl, authors, year)
        if rev:
            logger.info("doi_reverse_lookup doi=%s candidate=%r", rev, cand or None)
            return DoiVerifyResult(doi=rev, source="reverse-lookup")

    # 3. Nothing verified — keep the candidate as a hint, but mark it unverified.
    return DoiVerifyResult(doi=cand, source="unverified" if cand else "empty")
