"""XRD parser with Williamson-Hall + Scherrer + spectrum curve."""

from __future__ import annotations

import logging
import math
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from src.parsers._utils import downsample_curve

logger = logging.getLogger(__name__)

# Cu K-alpha1 wavelength in Angstroms (most common XRD source)
CU_KA1_ANGSTROM = 1.5406
K_SCHERRER = 0.94


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
                if 1 < x.min() < 80 and x.max() < 180:
                    return x, y
        except Exception:  # noqa: BLE001
            continue
    raise ValueError("Could not parse two-column XRD data")


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
        fwhm = float(widths[i]) * dx
        peaks.append({
            "two_theta": float(round(x[idx], 3)),
            "intensity": float(round(y_smooth[idx], 2)),
            "fwhm": float(round(fwhm, 4)),
            "relative_intensity": float(round(y_smooth[idx] / y_max * 100, 1)),
        })
    return peaks


def _scherrer_crystallite_size(peaks: list[dict[str, float]], wavelength: float = CU_KA1_ANGSTROM) -> float | None:
    """Average Scherrer crystallite size (nm) from top 3 peaks."""
    if len(peaks) < 3:
        return None
    top3 = sorted(peaks, key=lambda p: -p["relative_intensity"])[:3]
    sizes = []
    for p in top3:
        theta_rad = math.radians(p["two_theta"] / 2.0)
        fwhm_rad = math.radians(p["fwhm"])
        if fwhm_rad <= 0:
            continue
        D = (K_SCHERRER * wavelength) / (fwhm_rad * math.cos(theta_rad))  # in Å
        sizes.append(D / 10.0)  # → nm
    if not sizes:
        return None
    return float(round(sum(sizes) / len(sizes), 2))


def _williamson_hall(peaks: list[dict[str, float]], wavelength: float = CU_KA1_ANGSTROM) -> dict[str, Any] | None:
    """Williamson-Hall: βcosθ vs 4sinθ, slope = strain, intercept = Kλ/D."""
    if len(peaks) < 5:
        return None
    x_vals, y_vals = [], []
    for p in peaks:
        theta_rad = math.radians(p["two_theta"] / 2.0)
        fwhm_rad = math.radians(p["fwhm"])
        if fwhm_rad <= 0:
            continue
        x_vals.append(4.0 * math.sin(theta_rad))
        y_vals.append(fwhm_rad * math.cos(theta_rad))
    if len(x_vals) < 5:
        return None

    x_arr = np.array(x_vals)
    y_arr = np.array(y_vals)
    slope, intercept = np.polyfit(x_arr, y_arr, 1)
    y_pred = slope * x_arr + intercept
    ss_res = np.sum((y_arr - y_pred) ** 2)
    ss_tot = np.sum((y_arr - y_arr.mean()) ** 2)
    r_squared = float(round(1 - ss_res / ss_tot, 3)) if ss_tot > 0 else 0.0

    if intercept <= 0:
        return None
    D_angstrom = (K_SCHERRER * wavelength) / intercept
    return {
        "crystallite_size_nm": float(round(D_angstrom / 10.0, 2)),
        "microstrain": float(round(slope, 6)),
        "r_squared": r_squared,
        "method": "Williamson-Hall",
        "n_peaks_used": len(x_vals),
    }


def parse_xrd(raw_text: str, *, wavelength: float = CU_KA1_ANGSTROM) -> dict[str, Any]:
    x, y = _parse_two_column(raw_text)
    peaks = _detect_peaks(x, y)
    return {
        "spectrum_type": "xrd",
        "peaks": peaks,
        "spectrum_curve": downsample_curve(x, y, target_points=500),
        "quick_stats": {
            "rowCount": int(len(x)),
            "xRange": [float(round(x.min(), 2)), float(round(x.max(), 2))],
            "yRange": [float(round(y.min(), 2)), float(round(y.max(), 2))],
            "peakCount": len(peaks),
        },
        "scherrer_avg_nm": _scherrer_crystallite_size(peaks, wavelength),
        "williamson_hall": _williamson_hall(peaks, wavelength),
        "wavelength_angstrom": wavelength,
        "source": "Cu K-α₁",
    }
