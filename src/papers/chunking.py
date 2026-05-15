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

    # Sliding window
    chunks: list[Chunk] = []
    start = 0
    chunk_idx = 0
    total = len(char_stream)

    while start < total:
        end = min(start + TARGET_CHARS, total)
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
        start = end - OVERLAP_CHARS

    return chunks
