"""Parser dispatcher by spectrum type."""

from __future__ import annotations

from typing import Any, Callable

from src.parsers.ftir import parse_ftir
from src.parsers.raman import parse_raman
from src.parsers.uvvis import parse_uvvis
from src.parsers.uvvis_drs import parse_uvvis_drs
from src.parsers.xrd import parse_xrd

PARSERS: dict[str, Callable[[str], dict[str, Any]]] = {
    "xrd": parse_xrd,
    "uvvis": parse_uvvis,
    "uvvis_drs": parse_uvvis_drs,
    "raman": parse_raman,
    "ftir": parse_ftir,
}


def get_parser(spectrum_type: str) -> Callable[[str], dict[str, Any]]:
    parser = PARSERS.get(spectrum_type)
    if parser is None:
        raise ValueError(f"No parser for spectrum type: {spectrum_type}")
    return parser
