"""Raman parser with spectrum curve."""

from __future__ import annotations

import logging
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from src.parsers._utils import downsample_curve, normalize_decimal

logger = logging.getLogger(__name__)

D_BAND_RANGE = (1300, 1380)
G_BAND_RANGE = (1560, 1620)
TWOD_BAND_RANGE = (2650, 2750)


def _parse_two_column(text: str) -> tuple[np.ndarray, np.ndarray]:
    text = normalize_decimal(text)  # B4: EU decimal comma -> dot (no-op for US/dot)
    for sep in [",", ";", r"\s+", "\t"]:
        try:
            df = pd.read_csv(
                StringIO(text), sep=sep, header=None, comment="#",
                engine="python", skip_blank_lines=True,
            )
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            if df.shape[1] >= 2 and len(df) > 10:
                x = df.iloc[:, 0].to_numpy(dtype=float)
                y = df.iloc[:, 1].to_numpy(dtype=float)
                if x.min() >= 0 and x.max() < 5000:
                    return x, y
        except Exception:  # noqa: BLE001
            continue
    raise ValueError("Could not parse two-column Raman data")


def _detect_peaks(x: np.ndarray, y: np.ndarray, *, max_peaks: int = 30) -> list[dict[str, float]]:
    if len(y) >= 21:
        y_smooth = savgol_filter(y, window_length=11, polyorder=3)
    else:
        y_smooth = y
    prominence = (y_smooth.max() - y_smooth.min()) * 0.03
    peak_idx, props = find_peaks(y_smooth, prominence=prominence, distance=5, width=2)
    if len(peak_idx) > max_peaks:
        top = np.argsort(y_smooth[peak_idx])[-max_peaks:]
        peak_idx = peak_idx[np.sort(top)]
        props = {k: v[np.sort(top)] for k, v in props.items()}
    peaks = []
    widths = props.get("widths", np.zeros(len(peak_idx)))
    y_max = y_smooth.max()
    for i, idx in enumerate(peak_idx):
        dx = float(x[1] - x[0]) if len(x) > 1 else 0.0
        peaks.append({
            "shift_cm1": float(round(x[idx], 2)),
            "intensity": float(round(y_smooth[idx], 2)),
            "fwhm": float(round(float(widths[i]) * dx, 2)),
            "relative_intensity": float(round(y_smooth[idx] / y_max * 100, 1)),
        })
    return peaks


def _find_peak_in_range(peaks: list[dict[str, float]], rng: tuple[float, float]) -> dict[str, float] | None:
    in_rng = [p for p in peaks if rng[0] <= p["shift_cm1"] <= rng[1]]
    return max(in_rng, key=lambda p: p["intensity"]) if in_rng else None


def _carbon_analysis(peaks: list[dict[str, float]]) -> dict[str, Any] | None:
    d = _find_peak_in_range(peaks, D_BAND_RANGE)
    g = _find_peak_in_range(peaks, G_BAND_RANGE)
    twod = _find_peak_in_range(peaks, TWOD_BAND_RANGE)
    if not d or not g:
        return None
    id_ig = float(round(d["intensity"] / g["intensity"], 3))
    result: dict[str, Any] = {
        "d_band_cm1": d["shift_cm1"],
        "g_band_cm1": g["shift_cm1"],
        "id_ig_ratio": id_ig,
        "interpretation": (
            "Low disorder (high crystallinity)" if id_ig < 0.3
            else "Moderate disorder" if id_ig < 1.0
            else "High disorder (defects/amorphous)"
        ),
    }
    if twod:
        result["2d_band_cm1"] = twod["shift_cm1"]
        result["i2d_ig_ratio"] = float(round(twod["intensity"] / g["intensity"], 3))
    return result


def parse_raman(raw_text: str) -> dict[str, Any]:
    x, y = _parse_two_column(raw_text)
    peaks = _detect_peaks(x, y)
    return {
        "spectrum_type": "raman",
        "peaks": peaks,
        "spectrum_curve": downsample_curve(x, y, target_points=500),
        "quick_stats": {
            "rowCount": int(len(x)),
            "xRange": [float(round(x.min(), 1)), float(round(x.max(), 1))],
            "yRange": [float(round(y.min(), 2)), float(round(y.max(), 2))],
            "peakCount": len(peaks),
        },
        "carbon_analysis": _carbon_analysis(peaks),
        "x_unit": "cm-1",
        "y_unit": "Intensity (a.u.)",
    }
