"""
Raman scientific-analysis regression tests (R246).

Covers the upgraded analysis: wavelength-aware crystallite size (Cançado 2006),
integrated D/G area ratio, TMD (MoS2/WS2) layer-count, and curated band
assignment. Spectra are synthetic Gaussians at literature peak positions.
"""

from __future__ import annotations

import numpy as np

from src.parsers.raman import parse_raman


def _gauss(x: np.ndarray, center: float, amp: float, fwhm: float) -> np.ndarray:
    sigma = fwhm / 2.355
    return amp * np.exp(-((x - center) ** 2) / (2 * sigma**2))


def _to_text(x: np.ndarray, y: np.ndarray) -> str:
    return "\n".join(f"{xi:.1f}\t{yi:.2f}" for xi, yi in zip(x, y, strict=False))


def _carbon_spectrum() -> str:
    x = np.arange(100.0, 3000.0, 1.0)
    y = 50 + _gauss(x, 1350, 60, 45) + _gauss(x, 1580, 100, 30) + _gauss(x, 2690, 220, 32)
    return _to_text(x, y)


# --- wavelength-dependency (the key scientific guard) -----------------------

def test_la_none_without_wavelength() -> None:
    """No λ -> La must be None (ID/IG is excitation-dependent), with a note."""
    r = parse_raman(_carbon_spectrum())
    ca = r["carbon_analysis"]
    assert ca is not None
    assert ca["crystallite_size_la_nm"] is None
    assert any("wavelength" in n.lower() for n in ca["notes"])


def test_la_computed_with_wavelength() -> None:
    """With λ, La is a positive finite Cançado value."""
    r = parse_raman(_carbon_spectrum(), laser_wavelength=532.0)
    ca = r["carbon_analysis"]
    la = ca["crystallite_size_la_nm"]
    assert la is not None and la > 0
    assert "Cancado" in ca["la_method"]


def test_la_scales_with_wavelength_fourth_power() -> None:
    """La ∝ λ⁴ at fixed ID/IG: doubling-ish λ raises La sharply."""
    r532 = parse_raman(_carbon_spectrum(), laser_wavelength=532.0)
    r633 = parse_raman(_carbon_spectrum(), laser_wavelength=633.0)
    la532 = r532["carbon_analysis"]["crystallite_size_la_nm"]
    la633 = r633["carbon_analysis"]["crystallite_size_la_nm"]
    # (633/532)^4 ≈ 2.0
    assert la633 > la532
    assert abs((la633 / la532) - (633.0 / 532.0) ** 4) < 0.05


def test_tk_crosscheck_only_at_514() -> None:
    """Tuinstra-Koenig cross-check note appears only at ~514.5 nm."""
    r = parse_raman(_carbon_spectrum(), laser_wavelength=514.5)
    assert any("Tuinstra" in n for n in r["carbon_analysis"]["notes"])
    r2 = parse_raman(_carbon_spectrum(), laser_wavelength=785.0)
    assert not any("Tuinstra" in n for n in r2["carbon_analysis"]["notes"])


def test_area_and_height_ratios_both_present() -> None:
    ca = parse_raman(_carbon_spectrum(), laser_wavelength=532.0)["carbon_analysis"]
    assert ca["id_ig_ratio_area"] > 0
    assert ca["id_ig_ratio_height"] is not None


# --- TMD layer-count --------------------------------------------------------

def test_mos2_monolayer_separation() -> None:
    """MoS2 E2g≈384 / A1g≈403 -> Δ≈19 -> monolayer."""
    x = np.arange(100.0, 800.0, 0.5)
    y = 30 + _gauss(x, 384, 80, 6) + _gauss(x, 403, 90, 6)
    r = parse_raman(_to_text(x, y), laser_wavelength=532.0)
    mos2 = r["tmd_analysis"]["MoS2"]
    assert abs(mos2["separation_cm1"] - 19.0) < 2.0
    assert "monolayer" in mos2["layer_hint"]


def test_ws2_modes_detected() -> None:
    x = np.arange(100.0, 800.0, 0.5)
    y = 30 + _gauss(x, 351, 70, 7) + _gauss(x, 418, 100, 7)
    r = parse_raman(_to_text(x, y))
    assert "WS2" in r["tmd_analysis"]


# --- band assignment --------------------------------------------------------

def test_wo3_bands_assigned() -> None:
    """WO3 807/715/273 cm-1 annotated as W-O modes, not carbon."""
    x = np.arange(100.0, 1000.0, 0.5)
    y = 30 + _gauss(x, 807, 100, 12) + _gauss(x, 715, 80, 12) + _gauss(x, 273, 60, 10)
    r = parse_raman(_to_text(x, y))
    names = {b["material"] for b in r["band_assignments"]}
    assert "WO3" in names
    assert r["carbon_analysis"] is None


def test_no_bands_for_featureless() -> None:
    """A featureless ramp yields no spurious carbon analysis."""
    x = np.arange(100.0, 1000.0, 1.0)
    y = np.linspace(10.0, 12.0, len(x))
    r = parse_raman(_to_text(x, y))
    assert r["carbon_analysis"] is None
