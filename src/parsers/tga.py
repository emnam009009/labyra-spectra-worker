"""TGA (Thermogravimetric Analysis) parser.

Upgraded (R248): ISO 11358-1 extrapolated onset temperature per stage (tangent
∩ baseline — the reported standard, distinct from the DTG peak and from the
deviation onset), stability indices T5%/T10%, and explicit char yield (residue).

Input: 2-col (temperature_C or temperature_K, mass or mass_%)
Scientific methods: docs/scientific-methods/tga-analysis.md
"""

from __future__ import annotations

import logging
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from src.parsers._utils import downsample_curve, normalize_decimal

logger = logging.getLogger(__name__)


def _parse_two_column(text: str) -> tuple[np.ndarray, np.ndarray]:
    text = normalize_decimal(text)  # EU decimal comma -> dot
    for sep in [",", ";", r"\s+", "\t"]:
        try:
            df = pd.read_csv(
                StringIO(text), sep=sep, header=None, comment="#",
                engine="python", skip_blank_lines=True,
            )
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            if df.shape[1] >= 2 and len(df) > 20:
                x = df.iloc[:, 0].to_numpy(dtype=float)
                y = df.iloc[:, 1].to_numpy(dtype=float)
                if 20 < x.min() < 1500 and x.max() < 1500:
                    return x, y
        except Exception:
            continue
    raise ValueError("Could not parse two-column TGA data")


def _detect_temp_unit(x: np.ndarray) -> str:
    return "K" if x.min() > 200 and x.max() > 300 else "C"


def _detect_mass_mode(y: np.ndarray) -> str:
    if y.min() >= 0 and y.max() <= 110:
        return "percent"
    return "absolute"


def _to_percent(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "percent":
        return y
    return (y / y[0]) * 100.0


def _compute_dtg(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """DTG = -dm/dT (%/deg)."""
    y_smooth = savgol_filter(y, window_length=21, polyorder=3) if len(y) >= 21 else y
    return -np.gradient(y_smooth, x)


def _temperature_at_loss(x: np.ndarray, y_pct: np.ndarray, loss_pct: float) -> float | None:
    """Temperature at which cumulative mass loss first reaches loss_pct (e.g. T5%)."""
    target = y_pct[0] - loss_pct
    below = np.where(y_pct <= target)[0]
    if len(below) == 0:
        return None
    i = int(below[0])
    if i == 0:
        return float(round(x[0], 2))
    # linear interpolate between i-1 and i for sub-step precision
    x0, x1 = x[i - 1], x[i]
    y0, y1 = y_pct[i - 1], y_pct[i]
    if y1 == y0:
        return float(round(x1, 2))
    t = x0 + (target - y0) * (x1 - x0) / (y1 - y0)
    return float(round(t, 2))


def _extrapolated_onset(
    x: np.ndarray, y_pct: np.ndarray, dtg: np.ndarray, peak_idx: int, onset_idx: int
) -> float | None:
    """
    ISO 11358-1 extrapolated onset: intersection of the pre-step mass baseline
    and the tangent to the TGA curve at the point of maximum gradient (DTG peak).

      tangent:  m(T) = y[peak] + slope·(T - x[peak]),  slope = dm/dT = -DTG[peak]
      baseline: m = mean(y over the flat region just before the step)
      onset_T:  x[peak] + (baseline - y[peak]) / slope
    """
    slope = -dtg[peak_idx]  # dm/dT (negative during mass loss)
    if abs(slope) < 1e-9:
        return None
    lo = max(0, onset_idx - 10)
    baseline = float(np.mean(y_pct[lo:onset_idx + 1])) if onset_idx > lo else float(y_pct[onset_idx])
    onset_t = x[peak_idx] + (baseline - y_pct[peak_idx]) / slope
    # sanity: onset must be at/below the peak temperature and within range
    if onset_t < x.min() or onset_t > x[peak_idx] + 1.0:
        return None
    return float(round(onset_t, 2))


def _find_decomp_stages(
    x: np.ndarray, y_pct: np.ndarray, dtg: np.ndarray
) -> list[dict[str, float | None]]:
    dtg_smooth = savgol_filter(dtg, window_length=11, polyorder=2) if len(dtg) >= 11 else dtg
    prominence = max((dtg_smooth.max() - dtg_smooth.min()) * 0.05, 0.001)
    peak_idx, _ = find_peaks(dtg_smooth, prominence=prominence, distance=20)

    stages: list[dict[str, float | None]] = []
    for idx in peak_idx[:10]:
        thresh = 0.1 * dtg_smooth[idx]
        onset_idx = 0
        for back in range(idx, -1, -1):
            if dtg_smooth[back] < thresh:
                onset_idx = back
                break
        end_idx = len(dtg_smooth) - 1
        for fwd in range(idx, len(dtg_smooth)):
            if dtg_smooth[fwd] < thresh:
                end_idx = fwd
                break

        stages.append({
            "deviation_onset_T": float(round(x[onset_idx], 2)),   # ASTM-style first deflection
            "extrapolated_onset_T": _extrapolated_onset(x, y_pct, dtg_smooth, int(idx), onset_idx),
            "peak_T": float(round(x[idx], 2)),                    # DTG max = T of max rate (Td)
            "end_T": float(round(x[end_idx], 2)),
            "mass_loss_pct": float(round(y_pct[onset_idx] - y_pct[end_idx], 2)),
            "dtg_max": float(round(dtg_smooth[idx], 4)),
        })
    return stages


def parse_tga(raw_text: str) -> dict[str, Any]:
    x, y = _parse_two_column(raw_text)
    temp_unit = _detect_temp_unit(x)
    mass_mode = _detect_mass_mode(y)
    y_pct = _to_percent(y, mass_mode)
    dtg = _compute_dtg(x, y_pct)
    stages = _find_decomp_stages(x, y_pct, dtg)

    char_yield = float(round(y_pct[-1], 2))  # residual mass at the end of the run

    return {
        "spectrum_type": "tga",
        "peaks": [],
        "spectrum_curve": downsample_curve(x, y_pct, target_points=500),
        "dtg_curve": downsample_curve(x, dtg, target_points=500),
        "decomp_stages": stages,
        "stability": {
            "T5_pct": _temperature_at_loss(x, y_pct, 5.0),
            "T10_pct": _temperature_at_loss(x, y_pct, 10.0),
            "T50_pct": _temperature_at_loss(x, y_pct, 50.0),
        },
        "quick_stats": {
            "rowCount": len(x),
            "xRange": [float(round(x.min(), 1)), float(round(x.max(), 1))],
            "yRange": [float(round(y_pct.min(), 2)), float(round(y_pct.max(), 2))],
            "peakCount": len(stages),
        },
        "initial_mass_pct": float(round(y_pct[0], 2)),
        "final_mass_pct": char_yield,
        "char_yield_pct": char_yield,
        "total_loss_pct": float(round(y_pct[0] - y_pct[-1], 2)),
        "temp_unit": temp_unit,
        "x_unit": f"deg-{temp_unit}",
        "y_unit": "Mass (%)",
    }
