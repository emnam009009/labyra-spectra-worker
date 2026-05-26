"""Tests for the unified spectrum loader (src.parsers._tabular.load_spectrum).

Covers the cases that motivated unification (R259): vendor headers, header-name
column detection, EU decimals, wrong-column rejection via validate, and the
real CorrWare/ZPlot instrument files.
"""

from __future__ import annotations

import pytest

from src.parsers._tabular import load_spectrum


def _numeric_block(n: int = 40, x0: float = 10.0, step: float = 0.5) -> str:
    return "\n".join(f"{x0 + k * step:.4f},{100 + k}" for k in range(n))


def test_plain_csv() -> None:
    x, y = load_spectrum(_numeric_block())
    assert len(x) == 40
    assert x[0] == pytest.approx(10.0)


def test_header_name_detection() -> None:
    """Descriptive headers + index column: pick 2theta, not the index column."""
    rows = "\n".join(f"{k},{10 + k},{500 + k}" for k in range(30))
    text = "index,2theta,intensity\n" + rows
    x, y = load_spectrum(text)
    # must pick the 2theta column (10..39), not the index column (0..29)
    assert x.min() >= 10 and x.max() <= 40


def test_vendor_header_stripped() -> None:
    """CorrWare/ZPlot-style text preamble (no column names) is skipped."""
    header = (
        "CORRW ASCII\n  CorrWare for Windows: Version 3.5a\n"
        "  Open Circuit Potential (V): -1.103\n  Surface Area: 1\n"
    )
    x, y = load_spectrum(header + _numeric_block(n=40))
    assert len(x) == 40
    assert x[0] == pytest.approx(10.0)


def test_eu_decimal_comma() -> None:
    """EU locale: comma decimal with semicolon delimiter is normalised."""
    text = "\n".join(f"{10 + k};{1.5 + k:.1f}".replace(".", ",") for k in range(30))
    x, y = load_spectrum(text)
    assert len(x) == 30
    assert y[0] == pytest.approx(1.5)


def test_validate_rejects_wrong_range() -> None:
    """A validator encoding the physical range rejects a bad column layout."""
    # x in 10..30 but we demand FTIR range 300-5000 -> must fail
    with pytest.raises(ValueError):
        load_spectrum(_numeric_block(n=40),
                      validate=lambda x, y: 300 < x.min() < 5000)


def test_too_few_rows_raises() -> None:
    with pytest.raises(ValueError):
        load_spectrum("1,2\n3,4\n", min_rows=10)


def test_whitespace_delimited() -> None:
    text = "\n".join(f"{10 + k}   {100 + k}" for k in range(30))
    x, y = load_spectrum(text)
    assert len(x) == 30
