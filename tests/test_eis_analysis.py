"""
EIS analysis tests (R250).

Two-tier: model-free readout (Rs, Rct, Cdl, Warburg, j0) and the optional
equivalent-circuit fit. Synthetic spectra are generated with impedance.py from
known Randles parameters (Rs=20, Rct=100, Cdl=25 uF).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.parsers.eis import parse_eis

impedance = pytest.importorskip("impedance")
from impedance.models.circuits import CustomCircuit  # noqa: E402


def _randles_text(warburg: bool = False) -> str:
    if warburg:
        c = CustomCircuit("R0-p(R1,C1)-W1", initial_guess=[20, 100, 25e-6, 50])
    else:
        c = CustomCircuit("R0-p(R1,C1)", initial_guess=[20, 100, 25e-6])
    f = np.logspace(5, -2, 60)
    z = c.predict(f, use_initial=True)
    return "\n".join(f"{fi:.6g},{zr:.6f},{zi:.6f}" for fi, zr, zi in zip(f, z.real, z.imag, strict=False))


def _polar_text() -> str:
    c = CustomCircuit("R0-p(R1,C1)", initial_guess=[20, 100, 25e-6])
    f = np.logspace(5, -2, 60)
    z = c.predict(f, use_initial=True)
    zmod = np.abs(z)
    phase = np.degrees(np.angle(z))
    return "\n".join(f"{fi:.6g},{m:.6f},{p:.6f}" for fi, m, p in zip(f, zmod, phase, strict=False))


# --- model-free readout -----------------------------------------------------

def test_rs_readout() -> None:
    mf = parse_eis(_randles_text(), do_fit=False)["model_free"]
    assert abs(mf["Rs_ohm"] - 20.0) < 1.0


def test_rct_readout() -> None:
    mf = parse_eis(_randles_text(), do_fit=False)["model_free"]
    assert abs(mf["Rct_ohm"] - 100.0) < 5.0


def test_cdl_readout() -> None:
    mf = parse_eis(_randles_text(), do_fit=False)["model_free"]
    cdl_uf = mf["Cdl_F"] * 1e6
    assert 20 < cdl_uf < 30  # ~25 uF (model-free estimate)


# --- robustness -------------------------------------------------------------

def test_sign_flip_robustness() -> None:
    """Z'' supplied positive should still give the correct Rs/Rct."""
    c = CustomCircuit("R0-p(R1,C1)", initial_guess=[20, 100, 25e-6])
    f = np.logspace(5, -2, 60)
    z = c.predict(f, use_initial=True)
    flipped = "\n".join(
        f"{fi:.6g},{zr:.6f},{-zi:.6f}" for fi, zr, zi in zip(f, z.real, z.imag, strict=False)
    )
    mf = parse_eis(flipped, do_fit=False)["model_free"]
    assert abs(mf["Rs_ohm"] - 20.0) < 1.0
    assert abs(mf["Rct_ohm"] - 100.0) < 5.0


def test_polar_format() -> None:
    mf = parse_eis(_polar_text(), data_format="polar", do_fit=False)["model_free"]
    assert abs(mf["Rs_ohm"] - 20.0) < 1.0
    assert abs(mf["Rct_ohm"] - 100.0) < 5.0


def test_warburg_detected() -> None:
    r = parse_eis(_randles_text(warburg=True), do_fit=False)
    assert r["model_free"]["warburg_detected"] is True


def test_no_warburg_on_clean_arc() -> None:
    r = parse_eis(_randles_text(warburg=False), do_fit=False)
    assert r["model_free"]["warburg_detected"] is False


# --- exchange current density ----------------------------------------------

def test_j0_computed_with_area() -> None:
    mf = parse_eis(_randles_text(), area_cm2=0.5, n_electrons=2, do_fit=False)["model_free"]
    assert mf["exchange_current_density_A_cm2"] is not None
    assert mf["exchange_current_density_A_cm2"] > 0


def test_j0_none_without_area() -> None:
    r = parse_eis(_randles_text(), do_fit=False)
    assert r["model_free"]["exchange_current_density_A_cm2"] is None
    assert any("area" in n.lower() for n in r["notes"])


# --- equivalent-circuit fit -------------------------------------------------

def test_circuit_fit_recovers_parameters() -> None:
    """Seeded Randles fit should recover Rs and Rct on clean synthetic data."""
    r = parse_eis(_randles_text(), do_fit=True)
    fit = r["circuit_fit"]
    assert fit is not None and "parameters" in fit
    assert abs(fit["parameters"]["R0"] - 20.0) < 2.0
    assert abs(fit["parameters"]["R1"] - 100.0) < 5.0
    assert fit["chi_square"] < 0.5


# --- incomplete arc (real-data edge case) -----------------------------------

def test_incomplete_arc_flagged() -> None:
    """
    A spectrum truncated before the semicircle closes (apex at the lowest
    frequency) must flag arc_incomplete, withhold Cdl, and note Rct is a bound.
    """
    # high-frequency half of a Randles arc only (apex never reached going down)
    c = CustomCircuit("R0-p(R1,C1)", initial_guess=[5, 2000, 20e-6])
    f = np.logspace(5, 1, 40)  # stops at 10 Hz, above the ~4 Hz apex
    z = c.predict(f, use_initial=True)
    text = "\n".join(f"{fi:.6g},{zr:.6f},{zi:.6f}" for fi, zr, zi in zip(f, z.real, z.imag, strict=False))
    r = parse_eis(text, do_fit=False)
    assert r["model_free"]["arc_incomplete"] is True
    assert r["model_free"]["Cdl_F"] is None
    assert any("not closed" in n.lower() or "lower bound" in n.lower() for n in r["notes"])


def test_multicolumn_autodetect() -> None:
    """ZPlot/Gamry-style >3-column rows: Z'' is the negative column, Z' precedes it."""
    c = CustomCircuit("R0-p(R1,C1)", initial_guess=[20, 100, 25e-6])
    f = np.logspace(5, -2, 60)
    z = c.predict(f, use_initial=True)
    # columns: freq, Ampl(0.01), Bias(0), Time, Z', Z'', GD(0), Err(0), Range
    rows = []
    for i, (fi, zr, zi) in enumerate(zip(f, z.real, z.imag, strict=False)):
        rows.append(f"{fi:.6g}\t0.01\t0\t{i * 0.5:.3f}\t{zr:.6f}\t{zi:.6f}\t0\t0\t3")
    r = parse_eis("\n".join(rows), do_fit=False)
    assert abs(r["model_free"]["Rs_ohm"] - 20.0) < 1.0
    assert abs(r["model_free"]["Rct_ohm"] - 100.0) < 5.0
