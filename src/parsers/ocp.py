"""OCP (Open-Circuit Potential) parser.

Input: 2-col (time_s, potential_V_vs_ref)
- Detect equilibrium plateau (last 10-30%)
- Compute drift rate (mV/s in final region)
- Useful for: stability check before EIS, Voc for photoelectrochemistry
"""

from __future__ import annotations

import logging
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd

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
                # Time >= 0, potential typically -3 to +3 V
                if x.min() >= 0 and abs(y.min()) < 10 and abs(y.max()) < 10:
                    return x, y
        except Exception:
            continue
    raise ValueError("Could not parse two-column OCP data")


def _equilibrium_analysis(t: np.ndarray, v: np.ndarray) -> dict[str, float]:
    """Analyze final 20% of curve."""
    n = len(t)
    tail_start = int(n * 0.8)
    t_tail = t[tail_start:]
    v_tail = v[tail_start:]

    eq_potential = float(round(v_tail.mean(), 4))
    eq_std = float(round(v_tail.std(), 5))

    # Drift rate = linear slope of tail (V/s)
    if len(t_tail) >= 5:
        slope, _ = np.polyfit(t_tail, v_tail, 1)
        drift_mV_per_s = float(round(slope * 1000.0, 5))
    else:
        drift_mV_per_s = 0.0

    # Stability classification
    abs_drift = abs(drift_mV_per_s)
    if abs_drift < 0.01:
        stability = "stable"
    elif abs_drift < 0.1:
        stability = "drifting"
    else:
        stability = "unstable"

    return {
        "equilibrium_potential_V": eq_potential,
        "std_dev_V": eq_std,
        "drift_mV_per_s": drift_mV_per_s,
        "stability": stability,
        "tail_duration_s": float(round(t_tail[-1] - t_tail[0], 2)),
    }


def parse_ocp(raw_text: str) -> dict[str, Any]:
    t, v = _parse_two_column(raw_text)
    eq = _equilibrium_analysis(t, v)

    return {
        "spectrum_type": "ocp",
        "peaks": [],
        "spectrum_curve": downsample_curve(t, v, target_points=500),
        "equilibrium": eq,
        "quick_stats": {
            "rowCount": len(t),
            "xRange": [float(round(t.min(), 2)), float(round(t.max(), 2))],
            "yRange": [float(round(v.min(), 4)), float(round(v.max(), 4))],
            "peakCount": 0,
        },
        "duration_s": float(round(t[-1] - t[0], 2)),
        "x_unit": "s",
        "y_unit": "Potential (V vs ref)",
    }
