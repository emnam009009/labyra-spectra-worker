"""Parser dispatcher by spectrum type."""

from __future__ import annotations

from typing import Any, Callable

from src.parsers.xrd import parse_xrd

# Map spectrumType → parser. Add new types here (uvvis, raman, ftir in 3c).
PARSERS: dict[str, Callable[[str], dict[str, Any]]] = {
    "xrd": parse_xrd,
}


def get_parser(spectrum_type: str) -> Callable[[str], dict[str, Any]]:
    parser = PARSERS.get(spectrum_type)
    if parser is None:
        raise ValueError(f"No parser for spectrum type: {spectrum_type}")
    return parser
