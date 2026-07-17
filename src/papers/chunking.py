"""Sliding window chunker.

Port labyra-app/src/lib/ai/rag/pipeline/chunking.ts 1:1.

Strategy: ~1024 tokens/chunk, 100 token overlap, respects paragraph + page
+ section boundaries.

Char-based approximation:
  - English ~4 chars/token
  - Vietnamese ~3 chars/token
  - Conservative: 3.5 chars/token

Section detection:
  - Markdown headings (# Foo, ## Bar)
  - ALL CAPS short lines (3-80 chars, has at least one A-Z letter)

@phase R167-B2
"""
from __future__ import annotations

import math
import re

from src.papers.types import Chunk, OcrResult

TARGET_TOKENS = 1024
OVERLAP_TOKENS = 100
CHARS_PER_TOKEN = 3.5
TARGET_CHARS = int(TARGET_TOKENS * CHARS_PER_TOKEN)
OVERLAP_CHARS = int(OVERLAP_TOKENS * CHARS_PER_TOKEN)

_MD_HEADING_RE = re.compile(r"^#+\s*")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


def _detect_section(line: str) -> str | None:
    """Detect section heading from a line. Mirrors TS detectSection()."""
    trimmed = line.strip()
    if trimmed.startswith("#"):
        return _MD_HEADING_RE.sub("", trimmed)[:80]
    # ALL CAPS, 4-79 chars, has at least one A-Z letter, not punctuation-only
    if (
        3 < len(trimmed) < 80
        and trimmed == trimmed.upper()
        and any("A" <= c <= "Z" for c in trimmed)
    ):
        return trimmed
    return None


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs on double newline. Mirrors TS splitParagraphs()."""
    return [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]


def _approx_tokens(text: str) -> int:
    """Approximate token count by char count."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def chunk_paper(ocr_result: OcrResult) -> list[Chunk]:
    """Split paper into ~1024 token chunks with 100 token overlap.

    Respects paragraph + section boundaries when possible. Tracks which pages
    + section each chunk spans.

    Returns empty list if no extractable text (degenerate input, not error).
    """
    # Build flat char stream with page + section metadata per char
    # Match TS chunking.ts data shape: list of (ch, page, section) tuples
    char_stream: list[tuple[str, int, str]] = []
    current_section = ""

    for page in ocr_result.pages:
        paragraphs = _split_paragraphs(page.text)
        for para in paragraphs:
            first_line = para.split("\n", 1)[0]
            new_section = _detect_section(first_line)
            if new_section:
                current_section = new_section
            for ch in para:
                char_stream.append((ch, page.page_number, current_section))
            # Paragraph separator (2 newlines, matches TS)
            char_stream.append(("\n", page.page_number, current_section))
            char_stream.append(("\n", page.page_number, current_section))

    if not char_stream:
        return []

    # R556: snap both window edges to word boundaries.
    #
    # The window is counted in characters, so `char_stream[start:end]` and
    # `start = end - OVERLAP_CHARS` land wherever the count runs out — which in
    # English prose is inside a word about five times in six. That is how a chunk
    # comes to begin "l bandgap energy…" when the paper says "optical bandgap
    # energy": the "optica" is not trimmed downstream, it was never written.
    # Every citation drawn from such a chunk shows the damage.
    #
    # Bounded at 40 characters: past that we are not near a boundary, and a run
    # that long without whitespace is a formula or a URL, where the exact cut
    # matters less than not walking half a chunk hunting for a space.
    SNAP_LIMIT = 40

    def _is_word(i: int) -> bool:
        return 0 <= i < total and char_stream[i][0].isalnum()

    def _snap_forward(i: int) -> int:
        """Advance to the start of the next whole word."""
        if not _is_word(i) or not _is_word(i - 1):
            return i
        j = i
        while j < total and j - i < SNAP_LIMIT and _is_word(j):
            j += 1
        while j < total and j - i < SNAP_LIMIT and not _is_word(j):
            j += 1
        return i if j - i >= SNAP_LIMIT else j

    def _snap_back(i: int) -> int:
        """Retreat to the end of the last whole word."""
        if not _is_word(i - 1) or not _is_word(i):
            return i
        j = i
        while j > 0 and i - j < SNAP_LIMIT and _is_word(j - 1):
            j -= 1
        return i if i - j >= SNAP_LIMIT else j

    # Sliding window
    chunks: list[Chunk] = []
    start = 0
    chunk_idx = 0
    total = len(char_stream)

    while start < total:
        raw_end = min(start + TARGET_CHARS, total)
        end = raw_end if raw_end >= total else _snap_back(raw_end)
        # Snapping must never eat the window. A 120-character formula with no
        # whitespace makes _snap_back walk all the way to `start`, the window
        # becomes empty, `start` never advances and the loop spins forever. The
        # SNAP_LIMIT check inside the helper does not catch it — it measures
        # distance, and the distance here is under the limit. Guaranteeing
        # progress belongs where progress is decided, not where the boundary is
        # guessed.
        if end <= start:
            end = raw_end
        window = char_stream[start:end]
        text = "".join(c[0] for c in window).strip()

        if not text:
            start = end
            continue

        pages_set: set[int] = set()
        sections_set: set[str] = set()
        for _ch, page_num, section in window:
            pages_set.add(page_num)
            if section:
                sections_set.add(section)

        pages = sorted(pages_set)
        # First section encountered in window (matches TS Array.from(set)[0])
        section = next(iter(sections_set), "")

        chunks.append(Chunk(
            chunkIdx=chunk_idx,
            text=text,
            pages=pages,
            section=section,
            tokens=_approx_tokens(text),
        ))
        chunk_idx += 1

        if end >= total:
            break
        # Snap the next start too, or the overlap re-introduces the split the
        # end just avoided — it is the same arithmetic on the same counter.
        # And never stand still: if snapping cannot get past this window's
        # start, take the raw index. A chunker that loops forever on a 40-char
        # formula is a worse bug than a split word.
        next_start = _snap_forward(max(0, end - OVERLAP_CHARS))
        start = next_start if next_start > start else end

    return chunks
