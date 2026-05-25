"""DSC (Differential Scanning Calorimetry) parser.

Input: 2-col (temperature_C or _K, heat_flow_mW or W/g)
- Detect endothermic (Tm melting) and exothermic (Tc crystallization) peaks
- Tg glass transition: inflection point in baseline
- Sign convention: endo up or endo down — auto-detect
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
    text = normalize_decimal(text)  # B4: EU decimal comma -> dot (no-op for US/dot)
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
        except Exception:  # noqa: BLE001
            continue
    raise ValueError("Could not parse two-column DSC data")


def _detect_peaks_bidirectional(
    x: np.ndarray, y: np.ndarray
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Find both endothermic (negative) and exothermic (positive) peaks.
    Returns (endo_peaks, exo_peaks).
    """
    if len(y) >= 21:
        y_smooth = savgol_filter(y, window_length=11, polyorder=3)
    else:
        y_smooth = y

    # Exothermic = positive peaks
    prominence = max((y_smooth.max() - y_smooth.min()) * 0.05, 0.01)
    exo_idx, exo_props = find_peaks(y_smooth, prominence=prominence, distance=15, width=3)
    endo_idx, endo_props = find_peaks(-y_smooth, prominence=prominence, distance=15, width=3)

    def to_peak_list(idx_arr, props, sign: str) -> list[dict[str, float]]:
        widths = props.get("widths", np.zeros(len(idx_arr)))
        dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
        return [
            {
                "peak_T": float(round(x[i], 2)),
                "heat_flow": float(round(y_smooth[i], 4)),
                "fwhm": float(round(float(widths[k]) * dx, 2)),
                "direction": sign,
            }
            for k, i in enumerate(idx_arr[:5])  # max 5 each
        ]

    return to_peak_list(endo_idx, endo_props, "endothermic"), to_peak_list(
        exo_idx, exo_props, "exothermic"
    )


def _detect_glass_transition(x: np.ndarray, y: np.ndarray) -> dict[str, float] | None:
    """Tg = inflection point in baseline (max d²y/dx²).
    Look in lower temperature region (first half).
    """
    if len(y) < 50:
        return None
    half = len(y) // 2
    y_first = savgol_filter(y[:half], window_length=21, polyorder=3) if half >= 21 else y[:half]
    x_first = x[:half]

    # Second derivative
    d2 = np.gradient(np.gradient(y_first, x_first), x_first)
    # Tg = location of max |d2| in baseline (before main peaks)
    abs_d2 = np.abs(d2)
    if abs_d2.max() < np.median(abs_d2) * 5:
        return None
    tg_idx = int(np.argmax(abs_d2))
    if tg_idx < 10 or tg_idx > len(x_first) - 10:
        return None
    return {
        "tg": float(round(x_first[tg_idx], 2)),
        "delta_cp_approx": float(round(abs_d2[tg_idx], 5)),
    }


def parse_dsc(raw_text: str) -> dict[str, Any]:
    x, y = _parse_two_column(raw_text)
    endo, exo = _detect_peaks_bidirectional(x, y)
    tg = _detect_glass_transition(x, y)

    return {
        "spectrum_type": "dsc",
        "peaks": endo + exo,  # union for chart markers
        "endothermic_peaks": endo,
        "exothermic_peaks": exo,
        "glass_transition": tg,
        "spectrum_curve": downsample_curve(x, y, target_points=500),
        "quick_stats": {
            "rowCount": int(len(x)),
            "xRange": [float(round(x.min(), 1)), float(round(x.max(), 1))],
            "yRange": [float(round(y.min(), 4)), float(round(y.max(), 4))],
            "peakCount": len(endo) + len(exo),
        },
        "x_unit": "deg-C",
        "y_unit": "Heat flow (mW or W/g)",
    }
