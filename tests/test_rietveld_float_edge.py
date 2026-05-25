"""
Float-precision + classification regression tests (worker audit B7, B8).

B7  normalize_formula used `c == int(c)`, fragile under float accumulation:
    a stoichiometry of 0.9999999 would print "W0.9999999O3" instead of "WO3".
B8  _conductivity_type used `band_gap == 0.0` exact compare; Materials Project
    may return computational noise like 1e-16, misclassifying a metal as a
    semiconductor.

(B5 — Rietveld divergence note — is exercised by the existing rietveld suite +
runtime; it requires lmfit/pymatgen to drive a full refinement, so it is not
unit-tested here. The fix only adds a note, no numeric change.)
"""

from __future__ import annotations

from src.citation.formula import normalize_formula
from src.materials.mp_sync import _conductivity_type

# --- B7: formula count precision --------------------------------------------

def test_b7_clean_integer_counts() -> None:
    """Whole-number stoichiometry prints without decimals."""
    assert normalize_formula("Fe2O3") == "Fe2O3"
    assert normalize_formula("H2O") == "H2O"


def test_b7_float_noise_rounds_to_integer() -> None:
    """Count with float-accumulation noise prints as the clean integer."""
    # 2.0000001 must render as '2', not '2.0000001'
    assert normalize_formula("Fe2.0000001O3") == "Fe2O3"
    assert normalize_formula("Fe1.9999998O3") == "Fe2O3"


def test_b7_genuine_fraction_preserved() -> None:
    """A real non-integer stoichiometry is still shown (not rounded away)."""
    out = normalize_formula("Fe2.5O3")
    assert "2.5" in out


# --- B8: conductivity classification ----------------------------------------

def test_b8_exact_zero_is_metal() -> None:
    assert _conductivity_type(0.0) == "metal"


def test_b8_computational_noise_is_metal() -> None:
    """Tiny positive noise from MP API must classify as metal, not semiconductor."""
    assert _conductivity_type(1e-16) == "metal"
    assert _conductivity_type(-1e-16) == "metal"


def test_b8_real_bandgap_not_metal() -> None:
    """A genuine band gap is not metal."""
    assert _conductivity_type(2.8) != "metal"


def test_b8_none_is_unknown() -> None:
    assert _conductivity_type(None) == "unknown"
