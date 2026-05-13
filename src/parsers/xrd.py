"""XRD parser: 2θ/intensity → peaks + Williamson-Hall analysis.

Input formats: .xy, .csv, .txt (two columns: angle, intensity)
Output: dict with peaks, fwhm, optional crystallite size + microstrain.
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

# Cu Kα wavelength (Å) — most common XRD source
CU_KA_WAVELENGTH = 1.5406


def _parse_two_column(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Robust two-column parser. Skips headers, handles ',' / whitespace / tab."""
    # Try comma-separated first
    for sep in [",", r"\s+", "\t"]:
        try:
            df = pd.read_csv(
                StringIO(text),
                sep=sep,
                header=None,
                comment="#",
                engine="python",
                skip_blank_lines=True,
            )
            # Filter to numeric rows only (drop headers)
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            if df.shape[1] >= 2 and len(df) > 10:
                x = df.iloc[:, 0].to_numpy(dtype=float)
                y = df.iloc[:, 1].to_numpy(dtype=float)
                # Sanity: 2θ should be 5-90°
                if 0 < x.min() < 90 and x.max() < 180:
                    return x, y
        except Exception as exc:  # noqa: BLE001
            logger.debug("parse attempt with sep=%r failed: %s", sep, exc)
            continue
    raise ValueError("Could not parse two-column XRD data")


def _detect_peaks(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_peaks: int = 30,
) -> list[dict[str, float]]:
    """Smooth + find_peaks + FWHM estimate."""
    # Light smoothing only if enough points
    if len(y) >= 21:
        y_smooth = savgol_filter(y, window_length=11, polyorder=3)
    else:
        y_smooth = y

    # Prominence threshold: 5% of max
    prominence = (y_smooth.max() - y_smooth.min()) * 0.05
    distance = max(5, len(x) // 200)  # min separation in samples

    peak_idx, props = find_peaks(
        y_smooth,
        prominence=prominence,
        distance=distance,
        width=2,
    )

    # Sort by intensity desc, take top N
    if len(peak_idx) > max_peaks:
        top = np.argsort(y_smooth[peak_idx])[-max_peaks:]
        peak_idx = peak_idx[np.sort(top)]
        props = {k: v[np.sort(top)] for k, v in props.items()}

    peaks = []
    widths = props.get("widths", np.zeros(len(peak_idx)))
    for i, idx in enumerate(peak_idx):
        # FWHM in 2θ units: width (samples) × Δx
        dx = float(x[1] - x[0]) if len(x) > 1 else 0.0
        fwhm = float(widths[i]) * dx
        peaks.append(
            {
                "two_theta": float(round(x[idx], 4)),
                "intensity": float(round(y_smooth[idx], 2)),
                "fwhm": float(round(fwhm, 4)),
                "relative_intensity": float(round(y_smooth[idx] / y_smooth.max() * 100, 1)),
            }
        )
    return peaks


def _scherrer_size(fwhm_deg: float, two_theta_deg: float, k: float = 0.9) -> float:
    """Scherrer equation: τ = K·λ / (β·cosθ). Returns crystallite size in nm."""
    if fwhm_deg <= 0:
        return 0.0
    theta_rad = math.radians(two_theta_deg / 2.0)
    beta_rad = math.radians(fwhm_deg)
    size_a = (k * CU_KA_WAVELENGTH) / (beta_rad * math.cos(theta_rad))
    return float(round(size_a / 10.0, 2))  # Å → nm


def _williamson_hall(peaks: list[dict[str, float]]) -> dict[str, Any] | None:
    """W-H plot: β·cosθ vs 4·sinθ → slope = strain, intercept = K·λ/D.

    Requires ≥ 5 peaks for reliable fit.
    """
    if len(peaks) < 5:
        return None

    theta = np.array([math.radians(p["two_theta"] / 2.0) for p in peaks])
    beta = np.array([math.radians(p["fwhm"]) for p in peaks])

    x = 4 * np.sin(theta)
    y = beta * np.cos(theta)

    # Linear fit
    slope, intercept = np.polyfit(x, y, 1)

    # Derived parameters
    if intercept <= 0:
        return None
    crystallite_size_nm = round((0.9 * CU_KA_WAVELENGTH) / (intercept * 10), 2)
    microstrain = round(float(slope), 6)

    # R² for fit quality
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = float(round(1 - ss_res / ss_tot, 3)) if ss_tot > 0 else 0.0

    return {
        "crystallite_size_nm": crystallite_size_nm,
        "microstrain": microstrain,
        "r_squared": r_squared,
        "method": "Williamson-Hall",
        "n_peaks_used": len(peaks),
    }


def parse_xrd(raw_text: str) -> dict[str, Any]:
    """Main entry. Returns dict with peaks + quick_stats + optional W-H."""
    x, y = _parse_two_column(raw_text)
    peaks = _detect_peaks(x, y)

    # Scherrer per-peak (top 3 by intensity)
    top_peaks = sorted(peaks, key=lambda p: -p["intensity"])[:3]
    scherrer_sizes_nm = [
        _scherrer_size(p["fwhm"], p["two_theta"]) for p in top_peaks if p["fwhm"] > 0
    ]
    avg_scherrer_nm = (
        float(round(sum(scherrer_sizes_nm) / len(scherrer_sizes_nm), 2))
        if scherrer_sizes_nm
        else None
    )

    wh = _williamson_hall(peaks)

    return {
        "spectrum_type": "xrd",
        "peaks": peaks,
        "quick_stats": {
            "rowCount": int(len(x)),
            "xRange": [float(round(x.min(), 2)), float(round(x.max(), 2))],
            "yRange": [float(round(y.min(), 2)), float(round(y.max(), 2))],
            "peakCount": len(peaks),
        },
        "scherrer_avg_nm": avg_scherrer_nm,
        "williamson_hall": wh,
        "wavelength_angstrom": CU_KA_WAVELENGTH,
        "source": "Cu Kα (assumed)",
    }
