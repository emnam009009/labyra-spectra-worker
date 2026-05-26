"""
FTIR analysis regression tests (R249).

Covers simple ATR correction (penetration-depth scaling proportional to wavenumber),
atmospheric-band flagging (CO2/H2O), and that transmission-mode behaviour and
functional-group assignment are preserved. Synthetic Gaussian bands.
"""

from __future__ import annotations

import numpy as np

from src.parsers.ftir import parse_ftir


def _gauss(x: np.ndarray, center: float, amp: float, fwhm: float) -> np.ndarray:
    sigma = fwhm / 2.355
    return amp * np.exp(-((x - center) ** 2) / (2 * sigma**2))


def _spectrum() -> str:
    """Absorbance: M-O (600), C=O (1720), O-H (3400), atmospheric CO2 (2349)."""
    x = np.arange(400.0, 4000.0, 2.0)
    y = (0.05 + _gauss(x, 600, 0.8, 40) + _gauss(x, 1720, 0.6, 30)
         + _gauss(x, 3400, 0.5, 80) + _gauss(x, 2349, 0.3, 15))
    return "\n".join(f"{xi:.1f},{yi:.4f}" for xi, yi in zip(x, y, strict=False))


def _peak_abs(result: dict, lo: float, hi: float) -> float | None:
    hits = [p["absorbance"] for p in result["peaks"] if lo < p["wavenumber_cm1"] < hi]
    return hits[0] if hits else None


# --- ATR correction (penetration-depth scaling) -----------------------------

def test_atr_flag_set() -> None:
    r = parse_ftir(_spectrum(), mode="atr")
    assert r["atr_corrected"] is True
    assert r["sampling_mode"] == "atr"


def test_atr_raises_high_wavenumber_relative_to_low() -> None:
    """
    ATR over-weights low wavenumber; the xwavenumber correction must raise the high-wavenumber band relative
    to the low-wavenumber band compared with the uncorrected spectrum.
    """
    raw = parse_ftir(_spectrum())             # no correction
    atr = parse_ftir(_spectrum(), mode="atr")  # corrected
    low_raw, high_raw = _peak_abs(raw, 580, 620), _peak_abs(raw, 3350, 3450)
    low_c, high_c = _peak_abs(atr, 580, 620), _peak_abs(atr, 3350, 3450)
    assert None not in (low_raw, high_raw, low_c, high_c)
    # ratio high/low must increase after correction
    assert (high_c / low_c) > (high_raw / low_raw)


def test_transmission_mode_not_corrected() -> None:
    r = parse_ftir(_spectrum(), mode="transmission")
    assert r["atr_corrected"] is False


def test_unknown_mode_warns() -> None:
    r = parse_ftir(_spectrum())  # mode=None
    assert r["atr_corrected"] is False
    assert any("mode" in n.lower() for n in r["notes"])


# --- atmospheric flagging ---------------------------------------------------

def test_co2_flagged() -> None:
    r = parse_ftir(_spectrum())
    species = {b["species"] for b in r["atmospheric_bands"]}
    assert "CO2" in species


def test_co2_band_position() -> None:
    r = parse_ftir(_spectrum())
    co2 = [b for b in r["atmospheric_bands"] if b["species"] == "CO2"]
    assert co2
    assert any(2300 < w < 2380 for w in co2[0]["matched_peaks_cm1"])


# --- preserved behaviour ----------------------------------------------------

def test_functional_groups_preserved() -> None:
    r = parse_ftir(_spectrum())
    names = {g["name"] for g in r["functional_groups"]}
    assert "O-H stretch" in names
    assert "C=O stretch" in names


def test_transmittance_input_converts() -> None:
    """%T input still detected and converted (peaks found in absorbance domain)."""
    x = np.arange(400.0, 4000.0, 2.0)
    t = 100.0 - 60 * np.exp(-((x - 1720) ** 2) / (2 * (30 / 2.355) ** 2))  # dip = band
    text = "\n".join(f"{xi:.1f},{ti:.3f}" for xi, ti in zip(x, t, strict=False))
    r = parse_ftir(text)
    assert r["y_mode"] == "transmittance"
    assert r["quick_stats"]["peakCount"] >= 1
