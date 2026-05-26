"""UV-Vis parser with 4 Tauc transition types.

Transitions:
  - direct_allowed (gamma=1/2, exponent n=2)
  - direct_forbidden (gamma=3/2, exponent n=2/3)
  - indirect_allowed (gamma=2, exponent n=1/2)
  - indirect_forbidden (gamma=3, exponent n=1/3)

Pick best R^2 across all 4. Tauc plot exposes data for any chosen transition.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from src.parsers._utils import downsample_curve, load_xy

logger = logging.getLogger(__name__)

# (label, exponent n in (alpha*hv)^n formula, gamma value)
TRANSITIONS: list[tuple[str, float, str]] = [
    ("direct_allowed", 2.0, "1/2"),
    ("direct_forbidden", 2.0 / 3.0, "3/2"),
    ("indirect_allowed", 0.5, "2"),
    ("indirect_forbidden", 1.0 / 3.0, "3"),
]


def _parse_two_column(text: str) -> tuple[np.ndarray, np.ndarray]:
    return load_xy(
        text,
        validate=lambda x, y: 100 < x.min() < 1100 and x.max() < 2000,
        min_rows=10,
    )


def _tauc_bandgap_for_transition(
    energy_ev: np.ndarray,
    absorbance: np.ndarray,
    *,
    transition: str,
    n: float,
    gamma: str,
) -> dict[str, Any] | None:
    """Best-window linear fit on (alpha*hv)^n vs hv."""
    alpha_hv_n = (absorbance * energy_ev) ** n

    order = np.argsort(energy_ev)
    e_sorted = energy_ev[order]
    a_sorted = alpha_hv_n[order]

    if len(e_sorted) < 30:
        return None

    smoothed = savgol_filter(a_sorted, window_length=11, polyorder=2)
    baseline = float(np.percentile(smoothed[:len(smoothed) // 4], 50))
    peak_val = float(smoothed.max())
    threshold = baseline + 0.1 * (peak_val - baseline)
    above = np.where(smoothed > threshold)[0]
    if len(above) < 10:
        return None
    onset_idx = int(above[0])

    best = None
    best_score = 0.6
    for start_offset in range(0, min(30, len(smoothed) - onset_idx - 10)):
        i_start = onset_idx + start_offset
        for window_size in (8, 12, 16, 20, 25):
            i_end = i_start + window_size
            if i_end >= len(smoothed) - 2:
                continue
            x_fit = e_sorted[i_start:i_end]
            y_fit = smoothed[i_start:i_end]
            if len(x_fit) < 6:
                continue
            slope, intercept = np.polyfit(x_fit, y_fit, 1)
            if slope <= 0:
                continue
            y_pred = slope * x_fit + intercept
            ss_res = np.sum((y_fit - y_pred) ** 2)
            ss_tot = np.sum((y_fit - y_fit.mean()) ** 2)
            if ss_tot <= 0:
                continue
            r2 = float(1 - ss_res / ss_tot)
            bandgap = float(-intercept / slope)
            if bandgap < 0.5 or bandgap > 6.0:
                continue
            score = r2 * (1 + window_size / 50.0)
            if score > best_score:
                best_score = score
                best = {
                    "bandgap_ev": float(round(bandgap, 3)),
                    "transition": transition,
                    "gamma": gamma,
                    "exponent_n": float(round(n, 4)),
                    "r_squared": float(round(r2, 3)),
                    "fit_range_ev": [float(round(x_fit.min(), 3)), float(round(x_fit.max(), 3))],
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "method": f"Tauc plot — {transition} (n={round(n, 3)})",
                }
    return best


def _tauc_curve(energy_ev: np.ndarray, absorbance: np.ndarray, n: float) -> dict[str, list[float]]:
    """Compute (alpha*hv)^n vs hv for a specific n."""
    alpha_hv_n = (absorbance * energy_ev) ** n
    order = np.argsort(energy_ev)
    return downsample_curve(energy_ev[order], alpha_hv_n[order], target_points=400)


def _best_tauc(x: np.ndarray, y: np.ndarray) -> tuple[dict[str, Any] | None, str, float]:
    """Test all 4 transitions, return best fit + the n value used."""
    energy_ev = 1240.0 / x
    candidates: list[dict[str, Any]] = []
    for label, n_val, gamma in TRANSITIONS:
        result = _tauc_bandgap_for_transition(
            energy_ev, y, transition=label, n=n_val, gamma=gamma,
        )
        if result:
            candidates.append(result)

    if not candidates:
        # Return None bandgap but a curve for direct_allowed as default visualization
        return None, "direct_allowed", 2.0

    best = max(candidates, key=lambda c: c["r_squared"])
    return best, best["transition"], best["exponent_n"]


def _detect_absorption_peaks(
    wavelength_nm: np.ndarray, absorbance: np.ndarray, *, max_peaks: int = 10
) -> list[dict[str, float]]:
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
    x, y = _parse_two_column(raw_text)
    bandgap_result, best_label, best_n = _best_tauc(x, y)
    peaks = _detect_absorption_peaks(x, y)
    energy_ev = 1240.0 / x

    return {
        "spectrum_type": "uvvis",
        "peaks": peaks,
        "spectrum_curve": downsample_curve(x, y, target_points=500),
        "tauc_curve": _tauc_curve(energy_ev, y, best_n),
        "all_transition_fits": _all_transition_summary(energy_ev, y),
        "quick_stats": {
            "rowCount": len(x),
            "xRange": [float(round(x.min(), 1)), float(round(x.max(), 1))],
            "yRange": [float(round(y.min(), 4)), float(round(y.max(), 4))],
            "peakCount": len(peaks),
        },
        "tauc_bandgap": bandgap_result,
        "x_unit": "nm",
        "y_unit": "Absorbance",
    }


def _all_transition_summary(energy_ev: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
    """Return summary of all 4 transitions for transparency."""
    results = []
    for label, n_val, gamma in TRANSITIONS:
        res = _tauc_bandgap_for_transition(
            energy_ev, y, transition=label, n=n_val, gamma=gamma,
        )
        if res:
            results.append({
                "transition": label,
                "gamma": gamma,
                "bandgap_ev": res["bandgap_ev"],
                "r_squared": res["r_squared"],
            })
        else:
            results.append({
                "transition": label,
                "gamma": gamma,
                "bandgap_ev": None,
                "r_squared": None,
            })
    return results
