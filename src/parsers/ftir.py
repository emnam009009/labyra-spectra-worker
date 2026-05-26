"""FTIR parser with spectrum curve and sampling-mode awareness.

Upgraded (R249): simple ATR correction (penetration depth proportional to 1/wavenumber),
linear baseline subtraction for peak absorbance, and atmospheric-band flagging
(CO2 / H2O). Vendor header stripping (PerkinElmer, JCAMP-DX) and T↔A conversion
retained.

Scientific methods: docs/scientific-methods/ftir-analysis.md
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from src.parsers._utils import downsample_curve, load_xy

logger = logging.getLogger(__name__)

# Atmospheric interference windows (cm-1): CO2 asymmetric stretch + bend, and
# water vapour rotational/bending envelopes. Peaks here may be artefacts.
ATMOSPHERIC_BANDS = [
    {"range": (2300, 2380), "species": "CO2", "note": "atmospheric CO2 asymmetric stretch (~2349)"},
    {"range": (660, 670), "species": "CO2", "note": "atmospheric CO2 bend (~667)"},
    {"range": (3500, 3950), "species": "H2O", "note": "water-vapour stretch/rotational"},
    {"range": (1300, 1680), "species": "H2O", "note": "water-vapour bending (may overlap real bands)"},
]

ATR_REFERENCE_WAVENUMBER = 1000.0  # normalisation point for simple ATR correction


def _strip_pe_header(text: str) -> str:
    """Strip PerkinElmer Spectrum ASCII header (magic 'PE IR' + '#DATA' marker)."""
    if not text.startswith("PE IR"):
        return text
    data_idx = text.find("#DATA")
    if data_idx == -1:
        return text
    return text[data_idx + len("#DATA"):].lstrip()


def _strip_jcamp_header(text: str) -> str:
    """Strip JCAMP-DX header (lines starting with ##); data after ##XYDATA= etc."""
    if not text.lstrip().startswith("##"):
        return text
    lines = text.split("\n")
    data_start = -1
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("##XYDATA=") or s.startswith("##PEAK TABLE=") or s.startswith("##XYPOINTS="):
            data_start = i + 1
            break
    if data_start == -1:
        return text
    data_lines = [ln for ln in lines[data_start:] if not ln.strip().startswith("##")]
    return "\n".join(data_lines)


def _parse_two_column(text: str) -> tuple[np.ndarray, np.ndarray]:
    return load_xy(
        text,
        validate=lambda x, y: 300 < x.min() < 5000 and x.max() < 5000,
        min_rows=10,
    )


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


def _atr_correct(x: np.ndarray, y_abs: np.ndarray) -> np.ndarray:
    """
    Simple ATR correction (first approximation): penetration depth dp proportional to 1/wavenumber, so
    ATR over-weights low wavenumbers vs transmission. Multiply by wavenumber to compensate
    (normalised at 1000 cm⁻¹). Corrects penetration-depth scaling ONLY — it does
    NOT remove the anomalous-dispersion red-shift/band-shape change (that needs an
    advanced correction with the sample refractive index + Kramers-Kronig).
    Ref: Anton Paar ATR wiki; ScienceDirect ATR overview.
    """
    return y_abs * (x / ATR_REFERENCE_WAVENUMBER)


def _linear_baseline(y: np.ndarray) -> np.ndarray:
    """Subtract a straight baseline between the two spectrum ends."""
    n = len(y)
    if n < 2:
        return y
    baseline = np.linspace(y[0], y[-1], n)
    return y - baseline


def _detect_peaks(x: np.ndarray, y_abs: np.ndarray, *, max_peaks: int = 30) -> list[dict[str, float]]:
    # PerkinElmer ASC files have descending x (4000 -> 400). Sort ascending so
    # dx > 0 and FWHM stays positive.
    if len(x) > 1 and x[1] < x[0]:
        x = x[::-1].copy()
        y_abs = y_abs[::-1].copy()

    y_smooth = savgol_filter(y_abs, window_length=11, polyorder=3) if len(y_abs) >= 21 else y_abs
    y_bc = _linear_baseline(y_smooth)  # baseline-subtracted for peak heights
    prominence = (y_bc.max() - y_bc.min()) * 0.03
    peak_idx, props = find_peaks(y_bc, prominence=prominence, distance=5, width=2)
    if len(peak_idx) > max_peaks:
        top = np.argsort(y_bc[peak_idx])[-max_peaks:]
        peak_idx = peak_idx[np.sort(top)]
        props = {k: v[np.sort(top)] for k, v in props.items()}
    peaks = []
    widths = props.get("widths", np.zeros(len(peak_idx)))
    dx = abs(float(x[1] - x[0])) if len(x) > 1 else 0.0
    for i, idx in enumerate(peak_idx):
        peaks.append({
            "wavenumber_cm1": float(round(x[idx], 1)),
            "absorbance": float(round(y_bc[idx], 4)),
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
        matched = [p["wavenumber_cm1"] for p in peaks if lo <= p["wavenumber_cm1"] <= hi]
        if matched:
            matches.append({
                "name": group["name"],
                "note": group["note"],
                "range_cm1": [lo, hi],
                "matched_peaks_cm1": matched,
            })
    return matches


def _flag_atmospheric(peaks: list[dict[str, float]]) -> list[dict[str, Any]]:
    flags = []
    for band in ATMOSPHERIC_BANDS:
        lo, hi = band["range"]
        matched = [p["wavenumber_cm1"] for p in peaks if lo <= p["wavenumber_cm1"] <= hi]
        if matched:
            flags.append({
                "species": band["species"],
                "note": band["note"],
                "range_cm1": [lo, hi],
                "matched_peaks_cm1": matched,
            })
    return flags


def parse_ftir(raw_text: str, mode: str | None = None) -> dict[str, Any]:
    raw_text = _strip_pe_header(raw_text)
    raw_text = _strip_jcamp_header(raw_text)
    x, y = _parse_two_column(raw_text)
    y_mode = _detect_y_mode(y)
    y_abs = _to_absorbance(y, y_mode)

    notes: list[str] = []
    is_atr = bool(mode and mode.lower() == "atr")
    if is_atr:
        y_abs = _atr_correct(x, y_abs)
        notes.append(
            "Simple ATR correction applied (x wavenumber, penetration-depth only); "
            "anomalous-dispersion shift not corrected (needs advanced ATR + refractive index)."
        )
    elif mode is None:
        notes.append(
            "Sampling mode unknown. If ATR, low-wavenumber bands are over-weighted "
            "vs transmission; provide mode='atr' for correction."
        )

    peaks = _detect_peaks(x, y_abs)
    atmospheric = _flag_atmospheric(peaks)
    if atmospheric:
        notes.append("Possible atmospheric (CO2/H2O) bands flagged; verify against background.")

    return {
        "spectrum_type": "ftir",
        "peaks": peaks,
        "spectrum_curve": downsample_curve(x, y, target_points=500),  # raw values
        "quick_stats": {
            "rowCount": len(x),
            "xRange": [float(round(x.min(), 1)), float(round(x.max(), 1))],
            "yRange": [float(round(y.min(), 4)), float(round(y.max(), 4))],
            "peakCount": len(peaks),
        },
        "y_mode": y_mode,
        "sampling_mode": mode.lower() if mode else None,
        "atr_corrected": is_atr,
        "functional_groups": _identify_functional_groups(peaks),
        "atmospheric_bands": atmospheric,
        "notes": notes,
        "x_unit": "cm-1",
        "y_unit": "%T" if y_mode == "transmittance" else "Absorbance",
    }
