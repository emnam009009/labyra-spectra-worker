"""DSC (Differential Scanning Calorimetry) parser with thermal analysis.

Upgraded (R247): peak enthalpy ΔH by baseline-subtracted integration (needs
heating rate β — no silent default), polymer crystallinity from ΔH_melt minus
cold-crystallization exotherm, and Tg reported with method (ISO 11357).

Input: 2-col (temperature_C or _K, heat_flow_mW or W/g)
Scientific methods: docs/scientific-methods/dsc-analysis.md
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from src.parsers._utils import downsample_curve, load_xy

logger = logging.getLogger(__name__)

# Heat of fusion of 100%-crystalline polymer, ΔH°f (J/g), peer-reviewed values.
# Used only when the caller declares the polymer.
REFERENCE_ENTHALPY_100: dict[str, float] = {
    "PE": 293.0,    # Mirabella & Bafna 2002 (J. Polym. Sci. B 40, 1637)
    "HDPE": 293.0,
    "LDPE": 293.0,
    "PP": 207.0,    # isotactic PP (common literature value)
    "PET": 140.1,   # poly(ethylene terephthalate)
    "PA6": 230.0,   # nylon-6
    "PA66": 196.0,
    "PLA": 93.0,    # poly(lactic acid)
    "PEEK": 130.0,
}


def _parse_two_column(text: str) -> tuple[np.ndarray, np.ndarray]:
    return load_xy(
        text,
        validate=lambda x, y: 20 < x.min() < 1500 and x.max() < 1500,
        min_rows=20,
    )


def _peak_bounds(x: np.ndarray, y_dev: np.ndarray, peak_i: int) -> tuple[int, int]:
    """Walk out from a peak to where the (baseline-deviation) signal returns to ~0."""
    n = len(x)
    thresh = abs(y_dev[peak_i]) * 0.05
    lo = peak_i
    while lo > 0 and abs(y_dev[lo]) > thresh:
        lo -= 1
    hi = peak_i
    while hi < n - 1 and abs(y_dev[hi]) > thresh:
        hi += 1
    return lo, hi


def _integrate_enthalpy(
    x: np.ndarray,
    y: np.ndarray,
    peak_idx: int,
    heating_rate_c_min: float | None,
    sample_mass_mg: float | None,
    y_in_w_per_g: bool,
) -> tuple[float | None, list[str]]:
    """
    ΔH of one peak: integrate heat flow over time across the peak, on a linear
    baseline between the peak feet.

      ∫ HF dt = (1/β[°C/s]) · ∫ HF dT      (β converts the T-axis to time)

    Units: W/g -> J/g directly; mW -> mJ, then /mass(g) -> J/g (mass required).
    Without β, ΔH cannot be computed (no silent default). Returns (ΔH_J_per_g, notes).
    """
    notes: list[str] = []
    if heating_rate_c_min is None:
        notes.append(
            "Enthalpy not computed: heating rate unknown. ΔH = (1/β)·∫HF dT "
            "needs β (°C/min); provide heatingRate."
        )
        return None, notes

    lo, hi = _peak_bounds(x, y - np.interp(x, [x[0], x[-1]], [y[0], y[-1]]), peak_idx)
    if hi - lo < 3:
        return None, notes
    xseg, yseg = x[lo:hi + 1], y[lo:hi + 1]
    # linear baseline between the two feet
    baseline = np.interp(xseg, [xseg[0], xseg[-1]], [yseg[0], yseg[-1]])
    dev = yseg - baseline
    area_t = float(np.trapezoid(dev, xseg))      # (HF unit)·°C
    beta_c_per_s = heating_rate_c_min / 60.0
    energy = abs(area_t) / beta_c_per_s          # (HF unit)·s = W·s = J  (per unit basis)

    if y_in_w_per_g:
        return float(round(energy, 2)), notes    # J/g
    # heat flow is mW -> energy is mJ; divide by mass to get J/g
    if sample_mass_mg is None:
        notes.append(
            "ΔH in mJ only (per sample): heat flow is mW and sample mass unknown; "
            "provide sampleMass (mg) for J/g."
        )
        return float(round(energy, 2)), notes     # mJ
    mass_g = sample_mass_mg / 1000.0
    energy_j = energy / 1000.0  # mJ -> J
    return float(round(energy_j / mass_g, 2)), notes  # J/g


def _detect_peaks_bidirectional(
    x: np.ndarray,
    y: np.ndarray,
    heating_rate_c_min: float | None,
    sample_mass_mg: float | None,
    y_in_w_per_g: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    y_smooth = savgol_filter(y, window_length=11, polyorder=3) if len(y) >= 21 else y
    prominence = max((y_smooth.max() - y_smooth.min()) * 0.05, 0.01)
    exo_idx, exo_props = find_peaks(y_smooth, prominence=prominence, distance=15, width=3)
    endo_idx, endo_props = find_peaks(-y_smooth, prominence=prominence, distance=15, width=3)

    def to_peak_list(idx_arr: np.ndarray, props: dict, sign: str) -> list[dict[str, Any]]:
        widths = props.get("widths", np.zeros(len(idx_arr)))
        dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
        out: list[dict[str, Any]] = []
        for k, i in enumerate(idx_arr[:5]):
            dh, _notes = _integrate_enthalpy(
                x, y_smooth, int(i), heating_rate_c_min, sample_mass_mg, y_in_w_per_g
            )
            out.append({
                "peak_T": float(round(x[i], 2)),
                "heat_flow": float(round(y_smooth[i], 4)),
                "fwhm": float(round(float(widths[k]) * dx, 2)),
                "direction": sign,
                "enthalpy_j_per_g": dh,
            })
        return out

    return (
        to_peak_list(endo_idx, endo_props, "endothermic"),
        to_peak_list(exo_idx, exo_props, "exothermic"),
    )


def _detect_glass_transition(x: np.ndarray, y: np.ndarray) -> dict[str, Any] | None:
    """Tg = inflection point (max |d²y/dx²|) in the low-T baseline (ISO 11357 inflection)."""
    if len(y) < 50:
        return None
    half = len(y) // 2
    y_first = savgol_filter(y[:half], window_length=21, polyorder=3) if half >= 21 else y[:half]
    x_first = x[:half]
    d2 = np.gradient(np.gradient(y_first, x_first), x_first)
    abs_d2 = np.abs(d2)
    if abs_d2.max() < np.median(abs_d2) * 5:
        return None
    tg_idx = int(np.argmax(abs_d2))
    if tg_idx < 10 or tg_idx > len(x_first) - 10:
        return None
    return {
        "tg": float(round(x_first[tg_idx], 2)),
        "method": "inflection (ISO 11357-2)",
        "delta_cp_approx": float(round(abs_d2[tg_idx], 5)),
    }


def _crystallinity(
    endo: list[dict[str, Any]],
    exo: list[dict[str, Any]],
    polymer: str | None,
) -> dict[str, Any] | None:
    """
    Xc = (ΔH_melt - ΔH_cold_cryst) / ΔH°f(polymer).
    Needs a declared polymer with a reference ΔH°f and computed enthalpies.
    """
    if not polymer:
        return None
    ref = REFERENCE_ENTHALPY_100.get(polymer.upper())
    if ref is None:
        return None
    h_melt = sum(p["enthalpy_j_per_g"] for p in endo if p.get("enthalpy_j_per_g"))
    h_cc = sum(p["enthalpy_j_per_g"] for p in exo if p.get("enthalpy_j_per_g"))
    if h_melt <= 0:
        return None
    xc = (h_melt - h_cc) / ref * 100.0
    return {
        "polymer": polymer.upper(),
        "reference_enthalpy_100_j_per_g": ref,
        "delta_h_melt_j_per_g": float(round(h_melt, 2)),
        "delta_h_cold_cryst_j_per_g": float(round(h_cc, 2)),
        "crystallinity_percent": float(round(max(0.0, min(100.0, xc)), 1)),
    }


def parse_dsc(
    raw_text: str,
    heating_rate_c_min: float | None = None,
    sample_mass_mg: float | None = None,
    polymer: str | None = None,
    y_unit: str | None = None,
) -> dict[str, Any]:
    x, y = _parse_two_column(raw_text)
    # W/g if the declared unit says so; otherwise assume mW (needs mass for J/g).
    y_in_w_per_g = bool(y_unit and ("w/g" in y_unit.lower() or "w g" in y_unit.lower()))

    endo, exo = _detect_peaks_bidirectional(
        x, y, heating_rate_c_min, sample_mass_mg, y_in_w_per_g
    )
    tg = _detect_glass_transition(x, y)
    crystallinity = _crystallinity(endo, exo, polymer)

    notes: list[str] = []
    if heating_rate_c_min is None:
        notes.append("Provide heatingRate (°C/min) to compute peak enthalpies (ΔH).")
    elif not y_in_w_per_g and sample_mass_mg is None:
        notes.append("Heat flow assumed in mW; provide sampleMass (mg) for ΔH in J/g.")

    return {
        "spectrum_type": "dsc",
        "peaks": endo + exo,
        "endothermic_peaks": endo,
        "exothermic_peaks": exo,
        "glass_transition": tg,
        "crystallinity": crystallinity,
        "heating_rate_c_min": heating_rate_c_min,
        "sample_mass_mg": sample_mass_mg,
        "notes": notes,
        "spectrum_curve": downsample_curve(x, y, target_points=500),
        "quick_stats": {
            "rowCount": len(x),
            "xRange": [float(round(x.min(), 1)), float(round(x.max(), 1))],
            "yRange": [float(round(y.min(), 4)), float(round(y.max(), 4))],
            "peakCount": len(endo) + len(exo),
        },
        "x_unit": "deg-C",
        "y_unit": y_unit or "Heat flow (mW or W/g)",
    }
