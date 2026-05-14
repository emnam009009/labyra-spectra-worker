"""XRD parser with Williamson-Hall + Scherrer + spectrum curve."""

from __future__ import annotations

import logging
import math
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from src.parsers._tabular import parse_csv_two_column, parse_xlsx_two_column
from src.parsers._utils import downsample_curve

logger = logging.getLogger(__name__)

# Cu K-alpha1 wavelength in Angstroms (most common XRD source)
CU_KA1_ANGSTROM = 1.5406
K_SCHERRER = 0.94

# X-ray anode Kα1 wavelengths (Angstroms)
ANODE_WAVELENGTHS: dict[str, float] = {
    "Cu": 1.5406,
    "Mo": 0.70932,
    "Co": 1.78897,
    "Cr": 2.29100,
    "Fe": 1.93604,
    "Ag": 0.55941,
}


# Monochromator type → Kα1 fraction (rest is Kα2)
# Higher Kα1% = better monochromaticity (Ge111/Johansson are pure Kα1)
MONOCHROMATOR_PRESETS: dict[str, float] = {
    "none": 0.67,         # Standard tube, Kα1:Kα2 = 2:1
    "ni_filter": 0.75,    # Ni filter removes Kβ but not Kα2
    "graphite": 0.85,     # Common pyrolytic graphite monochromator
    "ge111": 0.99,        # Ge(111) Johansson — nearly pure Kα1
    "johansson": 0.99,    # Same as ge111
    "si220": 0.995,       # High-resolution synchrotron-grade
}


def resolve_effective_wavelength(anode: str | None, monochromator: str | None = None) -> float:
    """Effective Kα = weighted mean of Kα1 + Kα2 based on monochromator.

    Returns Å. For pure Kα1 use Ge(111) or higher.
    """
    if not anode:
        anode = "Cu"
    ka1 = ANODE_WAVELENGTHS.get(anode, CU_KA1_ANGSTROM)
    # Kα2 ≈ Kα1 * 1.0025 (within 0.25% for all anodes typically)
    ka2 = ka1 * 1.00247  # rough average ratio
    frac = MONOCHROMATOR_PRESETS.get(monochromator or "none", 0.67)
    return frac * ka1 + (1.0 - frac) * ka2


def resolve_wavelength(anode):
    if not anode:
        return CU_KA1_ANGSTROM
    return ANODE_WAVELENGTHS.get(anode, CU_KA1_ANGSTROM)



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
    # Gate: R²<0.5 means linear W-H model doesn't fit (likely multi-phase or anisotropic)
    is_reliable = r_squared >= 0.5
    return {
        "crystallite_size_nm": float(round(D_angstrom / 10.0, 2)),
        "microstrain": float(round(slope, 6)),
        "r_squared": r_squared,
        "method": "Williamson-Hall",
        "n_peaks_used": len(x_vals),
        "is_reliable": is_reliable,
        "quality_note": (
            None if is_reliable
            else f"Linear W-H fit poor (R²={r_squared:.2f}). Possible multi-phase, anisotropic strain, or instrumental broadening. Use Scherrer or Rietveld."
        ),
    }


def parse_xrd(raw_text: str, *, wavelength: float = CU_KA1_ANGSTROM, anode: str | None = None, monochromator: str | None = None) -> dict[str, Any]:
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


# ============================================================
# R160-spectra-4a: Citation lookup hook
# ============================================================
def parse_xrd_with_citation(
    raw_text: str,
    *,
    sample_label: str | None = None,
    chemical_formula: str | None = None,
    filename: str | None = None,
    anode: str | None = None,
    monochromator: str | None = None,
) -> dict:
    """Parse XRD + attach citation candidates if formula resolvable."""
    from src.citation.lookup import lookup_xrd_candidates

    parsed = parse_xrd(raw_text, anode=anode, monochromator=monochromator)
    user_peaks = parsed.get("peaks", [])
    if not user_peaks:
        return parsed

    citation_result = lookup_xrd_candidates(
        user_peaks,
        sample_label=sample_label,
        chemical_formula=chemical_formula,
        filename=filename,
    )
    parsed["citation"] = citation_result
    return parsed


# ============================================================
# R160-spectra-4a-hotfix1: Excel support
# ============================================================
def parse_xrd_bytes(raw_bytes: bytes, filename: str, anode: str | None = None, monochromator: str | None = None) -> dict[str, Any]:
    """Parse XRD from raw bytes. Routes to .xlsx parser if .xlsx, else text."""
    if filename.lower().endswith(".xlsx"):
        result = parse_xlsx_two_column(raw_bytes)
        if result is None:
            raise ValueError("Could not parse XLSX (no valid 2theta+intensity columns)")
        x, y = result
        wavelength = resolve_effective_wavelength(anode, monochromator)
        return _build_xrd_result(x, y, wavelength=wavelength)
    # Decode bytes as text
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1", errors="replace")
    return parse_xrd(text)


def _build_xrd_result(x: np.ndarray, y: np.ndarray, *, wavelength: float = CU_KA1_ANGSTROM) -> dict[str, Any]:
    """Build XRD result dict from x,y arrays (shared with parse_xrd)."""
    peaks = _detect_peaks(x, y)
    scherrer = _scherrer_crystallite_size(peaks, wavelength) if peaks else None
    wh = _williamson_hall(peaks, wavelength) if peaks else None
    return {
        "spectrum_type": "xrd",
        "peaks": peaks,
        "spectrum_curve": downsample_curve(x, y, target_points=500),
        "quick_stats": {
            "rowCount": int(len(x)),
            "xRange": [float(round(x.min(), 2)), float(round(x.max(), 2))],
            "yRange": [float(round(y.min(), 1)), float(round(y.max(), 1))],
            "peakCount": len(peaks),
        },
        "scherrer_avg_nm": float(round(scherrer, 2)) if scherrer else None,
        "williamson_hall": wh,
        "wavelength_angstrom": wavelength,
        "source": next((k for k, v in ANODE_WAVELENGTHS.items() if abs(v - wavelength) < 0.001), "Cu") + "-Kα1",
    }
