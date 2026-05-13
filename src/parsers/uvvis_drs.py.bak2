"""UV-Vis DRS parser: wavelength + diffuse reflectance → Kubelka-Munk + Tauc.

DRS = Diffuse Reflectance Spectroscopy, used for powders/opaque samples.
F(R) = (1-R)^2 / (2R)  — Kubelka-Munk function
Tauc plot on F(R) instead of absorbance.

Input format: 2-col (wavelength_nm, reflectance) — reflectance can be 0-1 or 0-100%.
"""

from __future__ import annotations

import logging
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from src.parsers._utils import downsample_curve

logger = logging.getLogger(__name__)


def _parse_two_column(text: str) -> tuple[np.ndarray, np.ndarray]:
    for sep in [",", r"\s+", "\t"]:
        try:
            df = pd.read_csv(
                StringIO(text), sep=sep, header=None, comment="#",
                engine="python", skip_blank_lines=True,
            )
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            if df.shape[1] >= 2 and len(df) > 10:
                x = df.iloc[:, 0].to_numpy(dtype=float)
                y = df.iloc[:, 1].to_numpy(dtype=float)
                if 100 < x.min() < 1100 and x.max() < 2000:
                    return x, y
        except Exception:  # noqa: BLE001
            continue
    raise ValueError("Could not parse two-column UV-Vis DRS data")


def _detect_reflectance_mode(y: np.ndarray) -> str:
    """%R (0-100) or fractional R (0-1)?"""
    if y.max() > 5:
        return "percent"
    return "fractional"


def _to_fractional(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "percent":
        return np.clip(y / 100.0, 1e-4, 1.0)
    return np.clip(y, 1e-4, 1.0)


def _kubelka_munk(reflectance: np.ndarray) -> np.ndarray:
    """F(R) = (1-R)^2 / (2R)."""
    return (1.0 - reflectance) ** 2 / (2.0 * reflectance)


def _tauc_bandgap_drs(
    wavelength_nm: np.ndarray,
    F_R: np.ndarray,
    *,
    transition: str = "direct",
) -> dict[str, Any] | None:
    """Tauc plot using Kubelka-Munk: (F(R)·hν)^n vs hν."""
    energy_ev = 1240.0 / wavelength_nm
    n = 2.0 if transition == "direct" else 0.5
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
    best_r2 = 0.6
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
            if score > best_r2:
                best_r2 = score
                best = {
                    "bandgap_ev": float(round(bandgap, 3)),
                    "transition": transition,
                    "r_squared": float(round(r2, 3)),
                    "fit_range_ev": [float(round(x_fit.min(), 3)), float(round(x_fit.max(), 3))],
                    "method": "Tauc plot on Kubelka-Munk F(R)",
                }
    return best


def _tauc_curve_drs(
    wavelength_nm: np.ndarray, F_R: np.ndarray, transition: str
) -> dict[str, list[float]]:
    energy_ev = 1240.0 / wavelength_nm
    n = 2.0 if transition == "direct" else 0.5
    fr_hv_n = (F_R * energy_ev) ** n
    order = np.argsort(energy_ev)
    return downsample_curve(energy_ev[order], fr_hv_n[order], target_points=400)


def parse_uvvis_drs(raw_text: str) -> dict[str, Any]:
    x, y_raw = _parse_two_column(raw_text)
    mode = _detect_reflectance_mode(y_raw)
    R = _to_fractional(y_raw, mode)
    F_R = _kubelka_munk(R)

    direct = _tauc_bandgap_drs(x, F_R, transition="direct")
    indirect = _tauc_bandgap_drs(x, F_R, transition="indirect")

    bandgap_result = None
    chosen_transition = "direct"
    if direct and indirect:
        if direct["r_squared"] >= indirect["r_squared"]:
            bandgap_result = direct
        else:
            bandgap_result = indirect
            chosen_transition = "indirect"
    elif direct:
        bandgap_result = direct
    elif indirect:
        bandgap_result = indirect
        chosen_transition = "indirect"

    return {
        "spectrum_type": "uvvis_drs",
        "peaks": [],  # DRS typically doesn't have sharp peaks
        "reflectance_curve": downsample_curve(x, R, target_points=500),
        "km_curve": downsample_curve(x, F_R, target_points=500),
        "tauc_curve": _tauc_curve_drs(x, F_R, chosen_transition),
        "quick_stats": {
            "rowCount": int(len(x)),
            "xRange": [float(round(x.min(), 1)), float(round(x.max(), 1))],
            "yRange": [float(round(R.min(), 4)), float(round(R.max(), 4))],
            "peakCount": 0,
        },
        "tauc_bandgap": bandgap_result,
        "reflectance_mode": mode,
        "x_unit": "nm",
        "y_unit": "Reflectance",
    }
