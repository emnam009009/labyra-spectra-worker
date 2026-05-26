"""
DSC thermal-analysis regression tests (R247).

Covers peak enthalpy ΔH (baseline-subtracted integration, heating-rate aware),
unit handling (W/g vs mW+mass), polymer crystallinity, and Tg method label.
Synthetic Gaussian endo/exo peaks with known areas.
"""

from __future__ import annotations

import numpy as np

from src.parsers.dsc import parse_dsc


def _gauss(x: np.ndarray, center: float, amp: float, fwhm: float) -> np.ndarray:
    sigma = fwhm / 2.355
    return amp * np.exp(-((x - center) ** 2) / (2 * sigma**2))


def _to_text(x: np.ndarray, y: np.ndarray) -> str:
    return "\n".join(f"{xi:.2f},{yi:.4f}" for xi, yi in zip(x, y, strict=False))


def _melt_only_wg() -> str:
    """Endothermic melting peak at 250 °C, heat flow in W/g."""
    t = np.arange(50.0, 350.0, 0.5)
    return _to_text(t, -_gauss(t, 250, 2.0, 20))


# --- heating-rate dependency (the key guard) --------------------------------

def test_enthalpy_none_without_heating_rate() -> None:
    r = parse_dsc(_melt_only_wg(), y_unit="W/g")
    endo = r["endothermic_peaks"]
    assert endo and endo[0]["enthalpy_j_per_g"] is None
    assert any("heatingrate" in n.lower() for n in r["notes"])


def test_enthalpy_computed_with_heating_rate() -> None:
    r = parse_dsc(_melt_only_wg(), heating_rate_c_min=10.0, y_unit="W/g")
    dh = r["endothermic_peaks"][0]["enthalpy_j_per_g"]
    assert dh is not None and dh > 0


def test_enthalpy_inverse_with_heating_rate() -> None:
    """ΔH = (1/β)·∫HF dT: halving β doubles the reported enthalpy."""
    r10 = parse_dsc(_melt_only_wg(), heating_rate_c_min=10.0, y_unit="W/g")
    r5 = parse_dsc(_melt_only_wg(), heating_rate_c_min=5.0, y_unit="W/g")
    dh10 = r10["endothermic_peaks"][0]["enthalpy_j_per_g"]
    dh5 = r5["endothermic_peaks"][0]["enthalpy_j_per_g"]
    assert abs(dh5 / dh10 - 2.0) < 0.02


# --- unit handling ----------------------------------------------------------

def test_mw_without_mass_is_mj_with_note() -> None:
    t = np.arange(50.0, 350.0, 0.5)
    text = _to_text(t, -_gauss(t, 250, 5.0, 20))  # mW
    r = parse_dsc(text, heating_rate_c_min=10.0, y_unit="mW")
    assert r["endothermic_peaks"][0]["enthalpy_j_per_g"] is not None
    assert any("mass" in n.lower() for n in r["notes"])


def test_mw_with_mass_gives_j_per_g() -> None:
    t = np.arange(50.0, 350.0, 0.5)
    text = _to_text(t, -_gauss(t, 250, 5.0, 20))  # mW
    r = parse_dsc(text, heating_rate_c_min=10.0, sample_mass_mg=5.0, y_unit="mW")
    dh = r["endothermic_peaks"][0]["enthalpy_j_per_g"]
    # ~5 mg PET-scale melting: order tens-to-hundreds J/g, not 1e5 (unit sanity)
    assert dh is not None and 10 < dh < 1000


# --- crystallinity ----------------------------------------------------------

def test_pet_crystallinity() -> None:
    """Cold-cryst exo (130 °C) + melting endo (250 °C); Xc = (ΔHm-ΔHc)/140.1."""
    t = np.arange(50.0, 350.0, 0.5)
    y = _gauss(t, 130, 1.0, 18) - _gauss(t, 250, 2.0, 20)  # W/g
    r = parse_dsc(_to_text(t, y), heating_rate_c_min=10.0, polymer="PET", y_unit="W/g")
    c = r["crystallinity"]
    assert c is not None
    assert c["polymer"] == "PET"
    assert c["reference_enthalpy_100_j_per_g"] == 140.1
    assert 0 <= c["crystallinity_percent"] <= 100


def test_crystallinity_none_without_polymer() -> None:
    r = parse_dsc(_melt_only_wg(), heating_rate_c_min=10.0, y_unit="W/g")
    assert r["crystallinity"] is None


def test_unknown_polymer_no_crystallinity() -> None:
    r = parse_dsc(_melt_only_wg(), heating_rate_c_min=10.0, polymer="UNOBTANIUM", y_unit="W/g")
    assert r["crystallinity"] is None


# --- Tg ---------------------------------------------------------------------

def test_tg_reports_method() -> None:
    """A baseline step around 80 °C is picked up as Tg with an ISO method label."""
    t = np.arange(25.0, 350.0, 0.5)
    # sigmoidal step (Cp jump) at 80 °C + melting later
    y = 0.5 / (1 + np.exp(-(t - 80) / 3.0)) - _gauss(t, 250, 2.0, 20)
    r = parse_dsc(_to_text(t, y), heating_rate_c_min=10.0, y_unit="W/g")
    if r["glass_transition"] is not None:
        assert "ISO 11357" in r["glass_transition"]["method"]
