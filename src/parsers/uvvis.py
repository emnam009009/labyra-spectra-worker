"""UV-Vis parser: wavelength (nm) + absorbance → Tauc bandgap + peaks.

Input format: .csv/.txt/.dpt 2-col (wavelength_nm, absorbance)
Common range: 200-800 nm (UV + visible)
"""

from __future__ import annotations

import logging
import math
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

logger = logging.getLogger(__name__)


def _parse_two_column(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Same robust parser as XRD."""
    for sep in [",", r"\s+", "\t"]:
        try:
            df = pd.read_csv(
                StringIO(text),
                sep=sep,
                header=None,
                comment="#",
                engine="python",
                skip_blank_lines=True,
            )
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            if df.shape[1] >= 2 and len(df) > 10:
                x = df.iloc[:, 0].to_numpy(dtype=float)
                y = df.iloc[:, 1].to_numpy(dtype=float)
                # UV-Vis wavelength range check (100-1100 nm reasonable)
                if 100 < x.min() < 1100 and x.max() < 2000:
                    return x, y
        except Exception:  # noqa: BLE001
            continue
    raise ValueError("Could not parse two-column UV-Vis data")


def _tauc_bandgap(
    wavelength_nm: np.ndarray,
    absorbance: np.ndarray,
    *,
    transition: str = "direct",
) -> dict[str, Any] | None:
    """Compute Tauc bandgap via linear extrapolation of (αhν)^n vs hν.

    transition: "direct" (n=2) or "indirect" (n=0.5).
    Returns bandgap in eV, linear fit R², and the energy range used.
    """
    # Convert: λ (nm) → photon energy E (eV)
    # E = 1240 / λ (nm)
    energy_ev = 1240.0 / wavelength_nm

    # α ∝ A (assume thin film, ignore reflection)
    # (α·hν)^n where n=2 for direct, 0.5 for indirect
    n = 2.0 if transition == "direct" else 0.5
    alpha_hv_n = (absorbance * energy_ev) ** n

    # Sort by energy ascending
    order = np.argsort(energy_ev)
    energy_sorted = energy_ev[order]
    alpha_sorted = alpha_hv_n[order]

    # Find the linear region: where derivative is maximal (steepest absorption edge)
    if len(energy_sorted) < 20:
        return None

    smoothed = savgol_filter(alpha_sorted, window_length=11, polyorder=2)
    derivative = np.gradient(smoothed, energy_sorted)

    # Find region of maximum derivative (the absorption edge)
    max_deriv_idx = int(np.argmax(derivative))
    if max_deriv_idx < 5 or max_deriv_idx > len(energy_sorted) - 5:
        return None

    # Use a window around the max derivative for linear fit
    window = min(10, max_deriv_idx, len(energy_sorted) - max_deriv_idx - 1)
    x_fit = energy_sorted[max_deriv_idx - window : max_deriv_idx + window]
    y_fit = alpha_sorted[max_deriv_idx - window : max_deriv_idx + window]

    if len(x_fit) < 4:
        return None

    # Linear fit: y = m·x + b → bandgap = -b/m (x-intercept)
    slope, intercept = np.polyfit(x_fit, y_fit, 1)
    if slope <= 0:
        return None
    bandgap = float(round(-intercept / slope, 3))

    if bandgap < 0.5 or bandgap > 6.0:
        return None  # unrealistic

    # R²
    y_pred = slope * x_fit + intercept
    ss_res = np.sum((y_fit - y_pred) ** 2)
    ss_tot = np.sum((y_fit - y_fit.mean()) ** 2)
    r_squared = float(round(1 - ss_res / ss_tot, 3)) if ss_tot > 0 else 0.0

    return {
        "bandgap_ev": bandgap,
        "transition": transition,
        "r_squared": r_squared,
        "fit_range_ev": [
            float(round(x_fit.min(), 3)),
            float(round(x_fit.max(), 3)),
        ],
        "method": "Tauc plot linear extrapolation",
    }


def _detect_absorption_peaks(
    wavelength_nm: np.ndarray,
    absorbance: np.ndarray,
    *,
    max_peaks: int = 10,
) -> list[dict[str, float]]:
    """Find absorption maxima."""
    if len(absorbance) >= 21:
        y_smooth = savgol_filter(absorbance, window_length=11, polyorder=3)
    else:
        y_smooth = absorbance

    prominence = (y_smooth.max() - y_smooth.min()) * 0.05
    peak_idx, _ = find_peaks(y_smooth, prominence=prominence, distance=10)

    if len(peak_idx) > max_peaks:
        top = np.argsort(y_smooth[peak_idx])[-max_peaks:]
        peak_idx = peak_idx[np.sort(top)]

    return [
        {
            "wavelength_nm": float(round(wavelength_nm[idx], 2)),
            "absorbance": float(round(y_smooth[idx], 4)),
            "energy_ev": float(round(1240.0 / wavelength_nm[idx], 3)),
        }
        for idx in peak_idx
    ]


def parse_uvvis(raw_text: str) -> dict[str, Any]:
    """Entry point. Returns parsed UV-Vis data."""
    x, y = _parse_two_column(raw_text)

    # Try both direct and indirect bandgap, pick higher R²
    direct = _tauc_bandgap(x, y, transition="direct")
    indirect = _tauc_bandgap(x, y, transition="indirect")

    bandgap_result = None
    if direct and indirect:
        bandgap_result = (
            direct if direct["r_squared"] >= indirect["r_squared"] else indirect
        )
    elif direct:
        bandgap_result = direct
    elif indirect:
        bandgap_result = indirect

    peaks = _detect_absorption_peaks(x, y)

    return {
        "spectrum_type": "uvvis",
        "peaks": peaks,
        "quick_stats": {
            "rowCount": int(len(x)),
            "xRange": [float(round(x.min(), 1)), float(round(x.max(), 1))],
            "yRange": [float(round(y.min(), 4)), float(round(y.max(), 4))],
            "peakCount": len(peaks),
        },
        "tauc_bandgap": bandgap_result,
        "x_unit": "nm",
        "y_unit": "Absorbance",
    }
