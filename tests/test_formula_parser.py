"""Test chemical formula parsing."""

from src.citation.formula import (
    elements_only,
    extract_formula_from_label,
    normalize_formula,
    parse_formula,
)


def test_parse_simple():
    assert parse_formula("WO3") == {"W": 1.0, "O": 3.0}


def test_parse_multi():
    assert parse_formula("Fe2O3") == {"Fe": 2.0, "O": 3.0}


def test_parse_decimal():
    result = parse_formula("W0.5O1.5")
    assert result["W"] == 0.5
    assert result["O"] == 1.5


def test_normalize_alphabetical():
    assert normalize_formula("O3W") == "O3W"  # already alphabetical (O < W)
    assert normalize_formula("Fe2O3") == "Fe2O3"


def test_elements_only():
    assert elements_only("Fe2O3") == ["Fe", "O"]
    assert elements_only("TiO2") == ["O", "Ti"]


def test_extract_from_label_simple():
    assert extract_formula_from_label("WO3-100C-1h") == "WO3"


def test_extract_from_label_complex():
    assert extract_formula_from_label("sample-A_TiO2_500C") == "TiO2"


def test_extract_from_label_none():
    assert extract_formula_from_label("unknown sample") is None
    assert extract_formula_from_label("") is None
    assert extract_formula_from_label(None) is None  # type: ignore[arg-type]
