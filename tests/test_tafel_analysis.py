"""Tests for the Tafel kinetics parser (src.parsers.tafel)."""

from __future__ import annotations

import numpy as np

from src.parsers.tafel import parse_tafel


def _her_curve(slope_v: float = 0.118, j0: float = 1e-3) -> str:
    """Synthetic HER polarization: eta = -slope*log10(j/j0), area = 1 cm2."""
    j = np.logspace(-2, 1.3, 60)          # 0.01 .. 20 mA/cm2
    eta = -slope_v * np.log10(j / j0)     # HER overpotential (negative)
    e_rhe = -eta                          # HER: eta = -E_RHE
    i_a = j / 1000.0                       # mA/cm2 -> A (area 1 cm2)
    return "\n".join(f"{e:.5f},{ii:.6e}" for e, ii in zip(e_rhe, i_a, strict=False))


def _oer_curve(slope_v: float = 0.060, j0: float = 1e-6) -> str:
    j = np.logspace(-2, 1.3, 60)
    eta = slope_v * np.log10(j / j0)
    e_rhe = eta + 1.23                     # OER: E_RHE = eta + 1.23
    i_a = j / 1000.0
    return "\n".join(f"{e:.5f},{ii:.6e}" for e, ii in zip(e_rhe, i_a, strict=False))


def test_her_slope_and_j0() -> None:
    r = parse_tafel(_her_curve(), reference="rhe", ph=0.0, area_cm2=1.0,
                    reaction="her")
    t = r["analysis"]["tafel"]
    assert abs(t["tafel_slope_mV_per_dec"] - 118.0) < 3.0
    assert abs(t["exchange_current_density_j0"] - 1e-3) < 5e-4
    assert abs(t["transfer_coefficient_alpha"] - 0.5) < 0.05
    assert t["r_squared"] > 0.99


def test_her_volmer_mechanism() -> None:
    r = parse_tafel(_her_curve(slope_v=0.118), reference="rhe", ph=0.0,
                    area_cm2=1.0, reaction="her")
    assert "Volmer" in r["analysis"]["tafel"]["mechanism_hint"]


def test_oer_not_single_step() -> None:
    r = parse_tafel(_oer_curve(), reference="rhe", ph=14.0, area_cm2=1.0,
                    reaction="oer")
    hint = r["analysis"]["tafel"]["mechanism_hint"]
    assert "multistep" in hint or "4-electron" in hint


def test_missing_reference_withholds_kinetics() -> None:
    r = parse_tafel(_her_curve(), reaction="her", area_cm2=1.0)
    assert "tafel" not in r["analysis"]
    assert any("reference" in n.lower() for n in r["notes"])


def test_area_unknown_flags_raw_j0() -> None:
    r = parse_tafel(_her_curve(), reference="rhe", ph=0.0, reaction="her")
    assert any("density" in n or "raw" in n for n in r["notes"])


def test_vendor_header_is_stripped() -> None:
    """Routes through load_xy, so a vendor preamble must not break parsing."""
    header = "GAMRY Framework\nExperiment: Tafel\nVCH: 0\n"
    r = parse_tafel(header + _her_curve(), reference="rhe", ph=0.0,
                    area_cm2=1.0, reaction="her")
    assert "tafel" in r["analysis"]
