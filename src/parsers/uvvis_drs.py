"""UV-Vis DRS parser with 4 Tauc transitions on Kubelka-Munk F(R)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from src.parsers._utils import downsample_curve, load_xy

logger = logging.getLogger(__name__)

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


def _detect_reflectance_mode(y: np.ndarray) -> str:
    return "percent" if y.max() > 5 else "fractional"


def _to_fractional(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "percent":
        return np.clip(y / 100.0, 1e-4, 1.0)
    return np.clip(y, 1e-4, 1.0)


def _kubelka_munk(R: np.ndarray) -> np.ndarray:
    return (1.0 - R) ** 2 / (2.0 * R)


def _tauc_bandgap_for_transition(
    energy_ev: np.ndarray,
    F_R: np.ndarray,
    *,
    transition: str,
    n: float,
    gamma: str,
) -> dict[str, Any] | None:
    fr_hv_n = (F_R * energy_ev) ** n
    order = np.argsort(energy_ev)
    e_sorted = energy_ev[order]
    fr_sorted = fr_hv_n[order]

    if len(e_sorted) < 30:
        return None

    smoothed = savgol_filter(fr_sorted, window_length=11, polyorder=2)
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
                    "method": f"Tauc on Kubelka-Munk F(R) — {transition} (n={round(n, 3)})",
                }
    return best


def _tauc_curve_drs(energy_ev: np.ndarray, F_R: np.ndarray, n: float) -> dict[str, list[float]]:
    fr_hv_n = (F_R * energy_ev) ** n
    order = np.argsort(energy_ev)
    return downsample_curve(energy_ev[order], fr_hv_n[order], target_points=400)


def _best_tauc_drs(x: np.ndarray, F_R: np.ndarray) -> tuple[dict[str, Any] | None, str, float]:
    energy_ev = 1240.0 / x
    candidates: list[dict[str, Any]] = []
    for label, n_val, gamma in TRANSITIONS:
        result = _tauc_bandgap_for_transition(
            energy_ev, F_R, transition=label, n=n_val, gamma=gamma,
        )
        if result:
            candidates.append(result)
    if not candidates:
        return None, "direct_allowed", 2.0
    best = max(candidates, key=lambda c: c["r_squared"])
    return best, best["transition"], best["exponent_n"]


def _all_transition_summary_drs(energy_ev: np.ndarray, F_R: np.ndarray) -> list[dict[str, Any]]:
    results = []
    for label, n_val, gamma in TRANSITIONS:
        res = _tauc_bandgap_for_transition(
            energy_ev, F_R, transition=label, n=n_val, gamma=gamma,
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


def parse_uvvis_drs(raw_text: str) -> dict[str, Any]:
    x, y_raw = _parse_two_column(raw_text)
    mode = _detect_reflectance_mode(y_raw)
    R = _to_fractional(y_raw, mode)
    F_R = _kubelka_munk(R)

    bandgap_result, _best_label, best_n = _best_tauc_drs(x, F_R)
    energy_ev = 1240.0 / x

    return {
        "spectrum_type": "uvvis_drs",
        "peaks": [],
        "reflectance_curve": downsample_curve(x, R, target_points=500),
        "km_curve": downsample_curve(x, F_R, target_points=500),
        "tauc_curve": _tauc_curve_drs(energy_ev, F_R, best_n),
        "all_transition_fits": _all_transition_summary_drs(energy_ev, F_R),
        "quick_stats": {
            "rowCount": len(x),
            "xRange": [float(round(x.min(), 1)), float(round(x.max(), 1))],
            "yRange": [float(round(R.min(), 4)), float(round(R.max(), 4))],
            "peakCount": 0,
        },
        "tauc_bandgap": bandgap_result,
        "reflectance_mode": mode,
        "x_unit": "nm",
        "y_unit": "Reflectance",
    }
