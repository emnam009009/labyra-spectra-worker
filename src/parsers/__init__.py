"""Parser dispatcher."""

from __future__ import annotations

from typing import Any, Callable

from src.parsers.cv import parse_cv
from src.parsers.dsc import parse_dsc
from src.parsers.eis import parse_eis
from src.parsers.ftir import parse_ftir
from src.parsers.lsv import parse_lsv
from src.parsers.ocp import parse_ocp
from src.parsers.raman import parse_raman
from src.parsers.tga import parse_tga
from src.parsers.uvvis import parse_uvvis
from src.parsers.uvvis_drs import parse_uvvis_drs
from src.parsers.xrd import parse_xrd

PARSERS: dict[str, Callable[[str], dict[str, Any]]] = {
    "xrd": parse_xrd,
    "uvvis": parse_uvvis,
    "uvvis_drs": parse_uvvis_drs,
    "raman": parse_raman,
    "ftir": parse_ftir,
    "tga": parse_tga,
    "dsc": parse_dsc,
    "ocp": parse_ocp,
    "lsv": parse_lsv,
    "cv": parse_cv,
    "eis": parse_eis,
}


def get_parser(spectrum_type: str) -> Callable[[str], dict[str, Any]]:
    parser = PARSERS.get(spectrum_type)
    if parser is None:
        raise ValueError(f"No parser for spectrum type: {spectrum_type}")
    return parser
