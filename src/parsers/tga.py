"""TGA (Thermogravimetric Analysis) parser.

Input: 2-col (temperature_C or temperature_K, mass or mass_%)
- Compute weight loss in stages (find inflection points via DTG)
- DTG = derivative thermogravimetry (-dm/dT)
- Report total weight loss, residue, decomposition stages
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
            if df.shape[1] >= 2 and len(df) > 20:
                x = df.iloc[:, 0].to_numpy(dtype=float)
                y = df.iloc[:, 1].to_numpy(dtype=float)
                # T range: 25-1000 C or 298-1273 K (typical TGA)
                if 20 < x.min() < 1500 and x.max() < 1500:
                    return x, y
        except Exception:  # noqa: BLE001
            continue
    raise ValueError("Could not parse two-column TGA data")


def _detect_temp_unit(x: np.ndarray) -> str:
    """K if min > 200 and max > 273; else C."""
    return "K" if x.min() > 200 and x.max() > 300 else "C"


def _detect_mass_mode(y: np.ndarray) -> str:
    """percent if 0-110, absolute (mg) otherwise."""
    if 0 <= y.min() and y.max() <= 110:
        return "percent"
    return "absolute"


def _to_percent(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "percent":
        return y
    # Normalize to initial mass
    return (y / y[0]) * 100.0


def _compute_dtg(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """DTG = -dm/dT (%/K or %/C)."""
    if len(y) >= 21:
        y_smooth = savgol_filter(y, window_length=21, polyorder=3)
    else:
        y_smooth = y
    return -np.gradient(y_smooth, x)


def _find_decomp_stages(
    x: np.ndarray, y_pct: np.ndarray, dtg: np.ndarray
) -> list[dict[str, float]]:
    """Each stage = DTG peak + before/after mass."""
    # Smooth DTG slightly
    if len(dtg) >= 11:
        dtg_smooth = savgol_filter(dtg, window_length=11, polyorder=2)
    else:
        dtg_smooth = dtg

    prominence = max((dtg_smooth.max() - dtg_smooth.min()) * 0.05, 0.001)
    peak_idx, _ = find_peaks(dtg_smooth, prominence=prominence, distance=20)

    stages = []
    for idx in peak_idx[:10]:  # max 10 stages
        # Look back for onset (where DTG starts rising significantly)
        onset_idx = idx
        for back in range(idx, max(0, idx - 50), -1):
            if dtg_smooth[back] < 0.1 * dtg_smooth[idx]:
                onset_idx = back
                break
        # Look forward for end
        end_idx = idx
        for fwd in range(idx, min(len(dtg_smooth) - 1, idx + 50)):
            if dtg_smooth[fwd] < 0.1 * dtg_smooth[idx]:
                end_idx = fwd
                break

        stages.append({
            "onset_T": float(round(x[onset_idx], 2)),
            "peak_T": float(round(x[idx], 2)),
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

    return {
        "spectrum_type": "tga",
        "peaks": [],
        "spectrum_curve": downsample_curve(x, y_pct, target_points=500),
        "dtg_curve": downsample_curve(x, dtg, target_points=500),
        "decomp_stages": stages,
        "quick_stats": {
            "rowCount": int(len(x)),
            "xRange": [float(round(x.min(), 1)), float(round(x.max(), 1))],
            "yRange": [float(round(y_pct.min(), 2)), float(round(y_pct.max(), 2))],
            "peakCount": len(stages),
        },
        "initial_mass_pct": float(round(y_pct[0], 2)),
        "final_mass_pct": float(round(y_pct[-1], 2)),
        "total_loss_pct": float(round(y_pct[0] - y_pct[-1], 2)),
        "temp_unit": temp_unit,
        "x_unit": f"deg-{temp_unit}",
        "y_unit": "Mass (%)",
    }
