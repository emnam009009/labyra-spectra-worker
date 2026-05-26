"""
LSV analysis tests (R253).

HER/OER benchmarks: overpotential at 10 mA/cm2, onset at 1 mA/cm2, Tafel slope
from the linear region, and RHE conversion. Synthetic Tafel-behaved
polarization curves with known slope and j0.
"""

from __future__ import annotations

import numpy as np

from src.parsers.lsv import parse_lsv

_B = 0.060      # Tafel slope V/dec
_J0 = 1e-3      # exchange current density mA/cm2


def _oer_text(reference: str = "ag/agcl", ph: float = 14.0) -> str:
    """OER: eta = b*log10(j/j0); E_RHE = eta + 1.23; back out measured potential."""
    j = np.logspace(-2, 1.5, 60)
    eta = _B * np.log10(j / _J0)
    e_rhe = eta + 1.23
    offset = {"ag/agcl": 0.197, "sce": 0.241}[reference]
    e_meas = e_rhe - offset - 0.059 * ph
    return "\n".join(f"{e:.5f},{c:.5f}" for e, c in zip(e_meas, j, strict=False))


def _her_text(ph: float = 0.0) -> str:
    """HER: cathodic, negative current, E_eq = 0 vs RHE."""
    j = -np.logspace(-2, 1.5, 60)
    eta = _B * np.log10(np.abs(j) / _J0)
    e_rhe = -eta
    e_meas = e_rhe - 0.197 - 0.059 * ph
    return "\n".join(f"{e:.5f},{c:.5f}" for e, c in zip(e_meas, j, strict=False))


# --- core benchmarks --------------------------------------------------------

def test_overpotential_at_10ma() -> None:
    r = parse_lsv(_oer_text(), area_cm2=1.0, reference="ag/agcl", ph=14, reaction="oer", ir_corrected=True)
    eta10 = r["analysis"]["overpotential_at_10mA_cm2_V"]
    assert abs(eta10 - 0.24) < 0.01  # b*log10(10/1e-3) = 0.24


def test_onset_potential() -> None:
    r = parse_lsv(_oer_text(), area_cm2=1.0, reference="ag/agcl", ph=14, reaction="oer", ir_corrected=True)
    onset = r["analysis"]["onset_overpotential_at_1mA_cm2_V"]
    assert abs(onset - 0.18) < 0.01  # b*log10(1/1e-3) = 0.18


def test_tafel_slope() -> None:
    r = parse_lsv(_oer_text(), area_cm2=1.0, reference="ag/agcl", ph=14, reaction="oer", ir_corrected=True)
    tafel = r["analysis"]["tafel"]
    assert tafel is not None
    assert abs(tafel["tafel_slope_mV_per_dec"] - 60.0) < 3.0
    assert tafel["r2"] > 0.99


def test_her_overpotential() -> None:
    r = parse_lsv(_her_text(), area_cm2=1.0, reference="ag/agcl", ph=0, reaction="her", ir_corrected=True)
    eta10 = r["analysis"]["overpotential_at_10mA_cm2_V"]
    assert abs(eta10 - 0.24) < 0.01


# --- RHE conversion / reference --------------------------------------------

def test_reference_offset_affects_eta() -> None:
    """Same data declared with SCE vs Ag/AgCl shifts eta by the offset difference."""
    r_agcl = parse_lsv(_oer_text("ag/agcl", 14), area_cm2=1.0, reference="ag/agcl", ph=14, reaction="oer", ir_corrected=True)
    # build SCE-referenced data, then read back with SCE
    r_sce = parse_lsv(_oer_text("sce", 14), area_cm2=1.0, reference="sce", ph=14, reaction="oer", ir_corrected=True)
    # both reconstruct the same eta (consistent round-trip)
    assert abs(r_agcl["analysis"]["overpotential_at_10mA_cm2_V"]
               - r_sce["analysis"]["overpotential_at_10mA_cm2_V"]) < 0.01


# --- guards (no silent assumptions) -----------------------------------------

def test_no_eta_without_reference() -> None:
    r = parse_lsv(_oer_text(), area_cm2=1.0, reaction="oer")  # no reference/pH
    assert "overpotential_at_10mA_cm2_V" not in r["analysis"]
    assert any("rhe" in n.lower() for n in r["notes"])


def test_no_benchmark_without_area() -> None:
    r = parse_lsv(_oer_text(), reference="ag/agcl", ph=14, reaction="oer")
    assert r["analysis"]["current_density_unit"] == "raw"
    assert any("area" in n.lower() for n in r["notes"])


def test_ir_warning_when_uncorrected() -> None:
    r = parse_lsv(_oer_text(), area_cm2=1.0, reference="ag/agcl", ph=14, reaction="oer", ir_corrected=False)
    assert any("ir" in n.lower() for n in r["notes"])


def test_unknown_reaction_warns() -> None:
    r = parse_lsv(_oer_text(), area_cm2=1.0, reference="ag/agcl", ph=14)
    assert any("reaction" in n.lower() for n in r["notes"])
