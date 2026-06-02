"""Normalize bibliographic text (titles, author + journal names) from PDFs, the
metadata LLM, Crossref and OpenAlex.

Publishers routinely emit Unicode look-alikes — U+2010 HYPHEN instead of the
ASCII '-', non-breaking hyphens, zero-width characters, NBSP, and JATS/HTML
markup or entities (e.g. <sub>, &amp;). Those render as "junk", break search/
matching, and look unprofessional. We fold them to clean text while PRESERVING
meaningful punctuation: en-dash (U+2013), em-dash (U+2014), minus sign (U+2212)
and accented / non-Latin letters (e.g. Vietnamese diacritics) are kept as-is.

Parity with labyra-app: src/lib/utils/normalize-text.ts (cleanText).

@phase R239-text-normalization (worker)
"""
from __future__ import annotations

import html
import re
import unicodedata

_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
# Hyphen look-alikes → ASCII '-'. Intentionally EXCLUDES en-dash (U+2013),
# em-dash (U+2014) and minus sign (U+2212), which carry meaning in titles and
# chemical formulae (e.g. "structure-property", "WO3-x").
_HYPHEN_LIKE_RE = re.compile("[\u2010\u2011\u2012\u2043\ufe58\ufe63\uff0d]")
_ZERO_WIDTH_RE = re.compile("[\u00ad\u200b\u200c\u200d\u2060\ufeff]")
_ODD_SPACE_RE = re.compile("[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")
_WS_RE = re.compile(r"\s+")


def clean_text(value: str | None) -> str:
    """Clean one bibliographic string. Returns '' for empty/whitespace input."""
    if not value:
        return ""
    s = unicodedata.normalize("NFC", value)
    s = html.unescape(s)  # decode &amp; &lt; &#x...; first (so encoded tags strip)
    s = _TAG_RE.sub("", s)
    s = _HYPHEN_LIKE_RE.sub("-", s)
    s = _ZERO_WIDTH_RE.sub("", s)
    s = _ODD_SPACE_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def clean_text_list(values: list[str] | None) -> list[str]:
    """Clean a list of strings, dropping any that become empty."""
    if not values:
        return []
    out = [clean_text(v) for v in values if isinstance(v, str)]
    return [v for v in out if v]
