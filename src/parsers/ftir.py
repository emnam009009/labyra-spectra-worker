"""FTIR parser: wavenumber (cm⁻¹) + %T or Absorbance → peaks + functional groups.

Input format: .csv/.dpt 2-col (wavenumber_cm-1, transmission_% or absorbance)
Common range: 400-4000 cm⁻¹
Detects %T vs A automatically based on value range.
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
                # FTIR wavenumber 400-4000 cm⁻¹
                if 300 < x.min() < 5000 and x.max() < 5000:
                    return x, y
        except Exception:  # noqa: BLE001
            continue
    raise ValueError("Could not parse two-column FTIR data")


def _detect_y_mode(y: np.ndarray) -> str:
    """Detect if y is %Transmission or Absorbance.

    %T: typically 0-100 (sometimes 0-1 if normalized)
    Absorbance: typically 0-3 (sometimes higher)
    """
    y_min, y_max = float(y.min()), float(y.max())
    if y_max > 20 and y_max <= 110:
        return "transmittance"
    if y_max < 5 and y_min >= -0.5:
        return "absorbance"
    return "unknown"


def _to_absorbance(y: np.ndarray, mode: str) -> np.ndarray:
    """Convert to absorbance for peak detection (peaks point UP)."""
    if mode == "transmittance":
        # A = -log10(T/100), guard against zero/negative
        y_safe = np.clip(y, 0.01, 100.0)
        return -np.log10(y_safe / 100.0)
    return y  # already absorbance


def _detect_peaks(
    x: np.ndarray, y_abs: np.ndarray, *, max_peaks: int = 30
) -> list[dict[str, float]]:
    if len(y_abs) >= 21:
        y_smooth = savgol_filter(y_abs, window_length=11, polyorder=3)
    else:
        y_smooth = y_abs

    prominence = (y_smooth.max() - y_smooth.min()) * 0.03
    peak_idx, props = find_peaks(y_smooth, prominence=prominence, distance=5, width=2)

    if len(peak_idx) > max_peaks:
        top = np.argsort(y_smooth[peak_idx])[-max_peaks:]
        peak_idx = peak_idx[np.sort(top)]
        props = {k: v[np.sort(top)] for k, v in props.items()}

    peaks = []
    widths = props.get("widths", np.zeros(len(peak_idx)))
    for i, idx in enumerate(peak_idx):
        dx = float(x[1] - x[0]) if len(x) > 1 else 0.0
        fwhm = float(widths[i]) * dx
        peaks.append({
            "wavenumber_cm1": float(round(x[idx], 1)),
            "absorbance": float(round(y_smooth[idx], 4)),
            "fwhm": float(round(fwhm, 1)),
        })
    return peaks


# Functional group fingerprint regions (cm⁻¹)
FUNCTIONAL_GROUPS = [
    {"range": (3200, 3600), "name": "O-H stretch", "note": "alcohols, water, carboxylic acid"},
    {"range": (3000, 3100), "name": "=C-H stretch", "note": "aromatic, alkene"},
    {"range": (2850, 2960), "name": "C-H stretch (sp3)", "note": "alkane"},
    {"range": (2200, 2280), "name": "C≡N stretch", "note": "nitrile"},
    {"range": (1680, 1750), "name": "C=O stretch", "note": "ketone, ester, carboxylic acid"},
    {"range": (1600, 1680), "name": "C=C stretch", "note": "alkene, aromatic"},
    {"range": (1500, 1600), "name": "N-H bend / aromatic C=C", "note": "amine, aromatic"},
    {"range": (1300, 1450), "name": "C-H bend", "note": "alkane"},
    {"range": (1050, 1300), "name": "C-O / C-N stretch", "note": "ether, ester, amine"},
    {"range": (600, 900), "name": "C-H out-of-plane bend", "note": "aromatic substitution"},
    {"range": (400, 700), "name": "Metal-O stretch", "note": "metal oxide (M-O)"},
]


def _identify_functional_groups(peaks: list[dict[str, float]]) -> list[dict[str, Any]]:
    """Match peaks to functional group fingerprint regions."""
    matches = []
    for group in FUNCTIONAL_GROUPS:
        lo, hi = group["range"]
        matched_peaks = [p for p in peaks if lo <= p["wavenumber_cm1"] <= hi]
        if matched_peaks:
            matches.append({
                "name": group["name"],
                "note": group["note"],
                "range_cm1": [lo, hi],
                "matched_peaks_cm1": [p["wavenumber_cm1"] for p in matched_peaks],
            })
    return matches


def parse_ftir(raw_text: str) -> dict[str, Any]:
    x, y = _parse_two_column(raw_text)
    y_mode = _detect_y_mode(y)
    y_abs = _to_absorbance(y, y_mode)

    peaks = _detect_peaks(x, y_abs)
    functional_groups = _identify_functional_groups(peaks)

    return {
        "spectrum_type": "ftir",
        "peaks": peaks,
        "quick_stats": {
            "rowCount": int(len(x)),
            "xRange": [float(round(x.min(), 1)), float(round(x.max(), 1))],
            "yRange": [float(round(y.min(), 4)), float(round(y.max(), 4))],
            "peakCount": len(peaks),
        },
        "y_mode": y_mode,  # "transmittance" | "absorbance" | "unknown"
        "functional_groups": functional_groups,
        "x_unit": "cm⁻¹",
        "y_unit": "%T" if y_mode == "transmittance" else "Absorbance",
    }
