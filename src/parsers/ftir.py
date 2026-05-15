"""FTIR parser with spectrum curve."""

from __future__ import annotations

import logging
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from src.parsers._utils import downsample_curve

logger = logging.getLogger(__name__)


def _strip_pe_header(text: str) -> str:
    """Strip PerkinElmer Spectrum ASCII header.

    PE format: magic line "PE IR" + ~25 header lines + "#DATA" marker
    + tab-separated wavenumber<TAB>%T data rows.

    Returns text starting from first data row. If not PE format, returns
    original text unchanged.

    @phase R163-worker-ftir-pe
    """
    if not text.startswith("PE IR"):
        return text
    # Find #DATA marker (case-sensitive per PE spec)
    data_idx = text.find("#DATA")
    if data_idx == -1:
        return text  # malformed, let parser try anyway
    # Skip "#DATA\n" or "#DATA\r\n"
    after_marker = text[data_idx + len("#DATA"):]
    # Strip leading newlines/whitespace
    return after_marker.lstrip()


def _strip_jcamp_header(text: str) -> str:
    """Strip JCAMP-DX header (lines starting with ##).

    JCAMP-DX is the IUPAC standard for spectral data. Header lines start
    with ##LABEL=value. Data section starts after ##XYDATA= or ##PEAK TABLE=.

    @phase R163-worker-ftir-jcamp
    """
    if not text.lstrip().startswith("##"):
        return text
    lines = text.split("\n")
    data_start = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("##XYDATA=") or stripped.startswith("##PEAK TABLE=") or stripped.startswith("##XYPOINTS="):
            data_start = i + 1
            break
    if data_start == -1:
        return text
    # Filter out trailing ##END= and any remaining ## lines
    data_lines = [l for l in lines[data_start:] if not l.strip().startswith("##")]
    return "\n".join(data_lines)


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
                if 300 < x.min() < 5000 and x.max() < 5000:
                    return x, y
        except Exception:  # noqa: BLE001
            continue
    raise ValueError("Could not parse two-column FTIR data")


def _detect_y_mode(y: np.ndarray) -> str:
    y_max = float(y.max())
    if 20 < y_max <= 110:
        return "transmittance"
    if y_max < 5:
        return "absorbance"
    return "unknown"


def _to_absorbance(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "transmittance":
        y_safe = np.clip(y, 0.01, 100.0)
        return -np.log10(y_safe / 100.0)
    return y


def _detect_peaks(x: np.ndarray, y_abs: np.ndarray, *, max_peaks: int = 30) -> list[dict[str, float]]:
    # R165-phase-3-ftir-fwhm: PerkinElmer ASC files have descending x (4000 → 400 cm⁻¹).
    # find_peaks expects monotonic-positive-step data; reverse if descending so
    # dx > 0 and FWHM stays positive. y_abs reindexed to match.
    if len(x) > 1 and x[1] < x[0]:
        x = x[::-1].copy()
        y_abs = y_abs[::-1].copy()

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
        # R165-phase-3-ftir-fwhm: abs() — defense even though we now sort ascending above
        dx = abs(float(x[1] - x[0])) if len(x) > 1 else 0.0
        peaks.append({
            "wavenumber_cm1": float(round(x[idx], 1)),
            "absorbance": float(round(y_smooth[idx], 4)),
            "fwhm": float(round(float(widths[i]) * dx, 1)),
        })
    return peaks


FUNCTIONAL_GROUPS = [
    {"range": (3200, 3600), "name": "O-H stretch", "note": "alcohols, water, carboxylic acid"},
    {"range": (3000, 3100), "name": "=C-H stretch", "note": "aromatic, alkene"},
    {"range": (2850, 2960), "name": "C-H stretch (sp3)", "note": "alkane"},
    {"range": (2200, 2280), "name": "C-triple-N stretch", "note": "nitrile"},
    {"range": (1680, 1750), "name": "C=O stretch", "note": "ketone, ester, carboxylic acid"},
    {"range": (1600, 1680), "name": "C=C stretch", "note": "alkene, aromatic"},
    {"range": (1500, 1600), "name": "N-H bend / aromatic C=C", "note": "amine, aromatic"},
    {"range": (1300, 1450), "name": "C-H bend", "note": "alkane"},
    {"range": (1050, 1300), "name": "C-O / C-N stretch", "note": "ether, ester, amine"},
    {"range": (600, 900), "name": "C-H out-of-plane bend", "note": "aromatic substitution"},
    {"range": (400, 700), "name": "Metal-O stretch", "note": "metal oxide (M-O)"},
]


def _identify_functional_groups(peaks: list[dict[str, float]]) -> list[dict[str, Any]]:
    matches = []
    for group in FUNCTIONAL_GROUPS:
        lo, hi = group["range"]
        matched = [p for p in peaks if lo <= p["wavenumber_cm1"] <= hi]
        if matched:
            matches.append({
                "name": group["name"],
                "note": group["note"],
                "range_cm1": [lo, hi],
                "matched_peaks_cm1": [p["wavenumber_cm1"] for p in matched],
            })
    return matches


def parse_ftir(raw_text: str) -> dict[str, Any]:
    # R163-worker-ftir-pe: strip vendor-specific headers before parsing
    raw_text = _strip_pe_header(raw_text)
    raw_text = _strip_jcamp_header(raw_text)
    x, y = _parse_two_column(raw_text)
    y_mode = _detect_y_mode(y)
    y_abs = _to_absorbance(y, y_mode)
    peaks = _detect_peaks(x, y_abs)
    return {
        "spectrum_type": "ftir",
        "peaks": peaks,
        "spectrum_curve": downsample_curve(x, y, target_points=500),  # raw values
        "quick_stats": {
            "rowCount": int(len(x)),
            "xRange": [float(round(x.min(), 1)), float(round(x.max(), 1))],
            "yRange": [float(round(y.min(), 4)), float(round(y.max(), 4))],
            "peakCount": len(peaks),
        },
        "y_mode": y_mode,
        "functional_groups": _identify_functional_groups(peaks),
        "x_unit": "cm-1",
        "y_unit": "%T" if y_mode == "transmittance" else "Absorbance",
    }
