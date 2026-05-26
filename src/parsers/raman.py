"""Raman parser with scientific analysis.

Upgraded (R246): wavelength-aware crystallite size (Cançado 2006), integrated
D/G area ratio (not just height), TMD layer-count (MoS2/WS2 E2g-A1g splitting),
and band assignment from a curated table sourced from peer-reviewed literature.

Scientific methods: docs/scientific-methods/raman-analysis.md
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from src.parsers._utils import downsample_curve, load_xy

logger = logging.getLogger(__name__)

# Carbon band windows (cm-1)
D_BAND_RANGE = (1300, 1380)
G_BAND_RANGE = (1560, 1620)
TWOD_BAND_RANGE = (2650, 2750)

# Tuinstra-Koenig constant is only valid at 514.5 nm excitation.
TK_REFERENCE_NM = 514.5
TK_CONSTANT_NM = 4.4  # C(514.5 nm) in nm; Tuinstra & Koenig 1970 / Knight & White 1989

# Cançado 2006 general equation (Appl. Phys. Lett. 88, 163106; DOI 10.1063/1.2196057):
#   La (nm) = (2.4e-10) * λ^4 * (I_D / I_G)^-1,  λ in nm, integrated-intensity ratio.
CANCADO_PREFACTOR = 2.4e-10

# ── Band assignment table (curated from peer-reviewed literature) ────────────
# Each entry: window (cm-1), label, material, reference. A peak is annotated
# only when it falls inside a window — no inference beyond the table.
RAMAN_BANDS: list[dict[str, Any]] = [
    # Carbonaceous (sp2/sp3)
    {"range": (1300, 1380), "name": "D band", "material": "carbon",
     "note": "disorder/defect-induced (A1g breathing)", "ref": "Ferrari & Robertson 2000"},
    {"range": (1560, 1620), "name": "G band", "material": "carbon",
     "note": "sp2 E2g stretching", "ref": "Ferrari & Robertson 2000"},
    {"range": (2650, 2750), "name": "2D band", "material": "carbon",
     "note": "second-order; layer-count sensitive", "ref": "Ferrari 2006"},
    # MoS2 (2H) — ScienceDirect S2095927316305643
    {"range": (375, 387), "name": "E2g (MoS2)", "material": "MoS2",
     "note": "in-plane Mo-S; ~383 cm-1", "ref": "2H-MoS2 (Lee 2010)"},
    {"range": (403, 412), "name": "A1g (MoS2)", "material": "MoS2",
     "note": "out-of-plane S; ~408 cm-1", "ref": "2H-MoS2 (Lee 2010)"},
    # WS2 — Nature Sci. Rep. srep01755 / srep19476
    {"range": (345, 360), "name": "E2g / 2LA(M) (WS2)", "material": "WS2",
     "note": "in-plane + 2LA(M) overlap ~351 cm-1", "ref": "Berkdemir 2013"},
    {"range": (413, 423), "name": "A1g (WS2)", "material": "WS2",
     "note": "out-of-plane W-S ~418 cm-1; layer-count sensitive", "ref": "Berkdemir 2013"},
    # WO3 (monoclinic) — characteristic W-O modes
    {"range": (800, 820), "name": "W-O stretch (WO3)", "material": "WO3",
     "note": "O-W-O stretching ~807 cm-1", "ref": "m-WO3 (Daniel 1987)"},
    {"range": (700, 730), "name": "W-O stretch (WO3)", "material": "WO3",
     "note": "O-W-O stretching ~715 cm-1", "ref": "m-WO3 (Daniel 1987)"},
    {"range": (265, 285), "name": "W-O bend (WO3)", "material": "WO3",
     "note": "O-W-O bending ~273 cm-1", "ref": "m-WO3 (Daniel 1987)"},
    # Generic metal-oxide envelope (low specificity)
    {"range": (400, 700), "name": "Metal-O (M-O)", "material": "oxide",
     "note": "metal-oxide lattice modes", "ref": "general"},
]


def _parse_two_column(text: str) -> tuple[np.ndarray, np.ndarray]:
    return load_xy(
        text,
        validate=lambda x, y: x.min() >= 0 and x.max() < 5000,
        min_rows=10,
    )


def _detect_peaks(x: np.ndarray, y: np.ndarray, *, max_peaks: int = 30) -> list[dict[str, float]]:
    y_smooth = savgol_filter(y, window_length=11, polyorder=3) if len(y) >= 21 else y
    baseline = float(np.median(y_smooth))
    prominence = (y_smooth.max() - y_smooth.min()) * 0.03
    peak_idx, props = find_peaks(y_smooth, prominence=prominence, distance=5, width=2)
    if len(peak_idx) > max_peaks:
        top = np.argsort(y_smooth[peak_idx])[-max_peaks:]
        peak_idx = peak_idx[np.sort(top)]
        props = {k: v[np.sort(top)] for k, v in props.items()}
    peaks = []
    widths = props.get("widths", np.zeros(len(peak_idx)))
    y_max = y_smooth.max()
    dx = float(x[1] - x[0]) if len(x) > 1 else 0.0
    for i, idx in enumerate(peak_idx):
        peaks.append({
            "shift_cm1": float(round(x[idx], 2)),
            "intensity": float(round(y_smooth[idx] - baseline, 2)),  # baseline-subtracted height
            "raw_intensity": float(round(y_smooth[idx], 2)),
            "fwhm": float(round(float(widths[i]) * dx, 2)),
            "relative_intensity": float(round(y_smooth[idx] / y_max * 100, 1)),
        })
    return peaks


def _find_peak_in_range(
    peaks: list[dict[str, float]], rng: tuple[float, float]
) -> dict[str, float] | None:
    in_rng = [p for p in peaks if rng[0] <= p["shift_cm1"] <= rng[1]]
    return max(in_rng, key=lambda p: p["intensity"]) if in_rng else None


def _band_area(x: np.ndarray, y: np.ndarray, center: float, fwhm: float) -> float:
    """Integrated intensity over center ± 2·FWHM, baseline-subtracted (trapezoid)."""
    if fwhm <= 0:
        fwhm = 20.0  # fallback window if FWHM unresolved
    lo, hi = center - 2 * fwhm, center + 2 * fwhm
    mask = (x >= lo) & (x <= hi)
    if mask.sum() < 3:
        return 0.0
    xb, yb = x[mask], y[mask]
    baseline = float(min(yb[0], yb[-1]))
    area = float(np.trapezoid(np.clip(yb - baseline, 0, None), xb))
    return area


def _crystallite_size_la(
    id_ig_area: float, laser_wavelength: float | None
) -> tuple[float | None, str | None, list[str]]:
    """
    La via Cançado 2006 (needs λ). Returns (La_nm, method, notes).
    Without λ, La cannot be computed correctly (ID/IG ∝ λ^-4) — return None + note,
    never silently assume 514 nm.
    """
    notes: list[str] = []
    if id_ig_area <= 0:
        return None, None, notes
    if laser_wavelength is None:
        notes.append(
            "Crystallite size La not computed: laser wavelength unknown. "
            "ID/IG is excitation-dependent (∝ λ⁻⁴); provide laserWavelength (nm)."
        )
        return None, None, notes
    la = CANCADO_PREFACTOR * (laser_wavelength ** 4) / id_ig_area
    method = f"Cancado 2006 (λ={laser_wavelength:.1f} nm, integrated ID/IG)"
    # Cross-check with Tuinstra-Koenig only when at its valid excitation.
    if abs(laser_wavelength - TK_REFERENCE_NM) <= 2.0:
        notes.append(
            f"Tuinstra-Koenig cross-check (514.5 nm): "
            f"La≈{TK_CONSTANT_NM / id_ig_area:.1f} nm"
        )
    return float(round(la, 2)), method, notes


def _carbon_analysis(
    peaks: list[dict[str, float]],
    x: np.ndarray,
    y: np.ndarray,
    laser_wavelength: float | None,
) -> dict[str, Any] | None:
    d = _find_peak_in_range(peaks, D_BAND_RANGE)
    g = _find_peak_in_range(peaks, G_BAND_RANGE)
    twod = _find_peak_in_range(peaks, TWOD_BAND_RANGE)
    if not d or not g:
        return None

    # Height ratio (Tuinstra-Koenig convention) and integrated-area ratio (Cançado).
    id_ig_height = float(round(d["intensity"] / g["intensity"], 3)) if g["intensity"] > 0 else None
    d_area = _band_area(x, y, d["shift_cm1"], d["fwhm"])
    g_area = _band_area(x, y, g["shift_cm1"], g["fwhm"])
    id_ig_area = float(round(d_area / g_area, 3)) if g_area > 0 else 0.0

    la_nm, la_method, la_notes = _crystallite_size_la(id_ig_area, laser_wavelength)

    result: dict[str, Any] = {
        "d_band_cm1": d["shift_cm1"],
        "g_band_cm1": g["shift_cm1"],
        "id_ig_ratio": id_ig_area,  # backward-compat alias (= integrated-area ratio)
        "id_ig_ratio_height": id_ig_height,
        "id_ig_ratio_area": id_ig_area,
        "crystallite_size_la_nm": la_nm,
        "la_method": la_method,
        "interpretation": (
            "Low disorder (high crystallinity)" if id_ig_area < 0.3
            else "Moderate disorder" if id_ig_area < 1.0
            else "High disorder (defects/amorphous)"
        ),
        "notes": la_notes,
    }
    if twod:
        result["2d_band_cm1"] = twod["shift_cm1"]
        result["2d_fwhm_cm1"] = twod["fwhm"]
        i2d_ig = float(round(twod["intensity"] / g["intensity"], 3)) if g["intensity"] > 0 else None
        result["i2d_ig_ratio"] = i2d_ig
        # Monolayer graphene hint: sharp 2D (FWHM≲40) + I2D/IG>2 (Ferrari 2006).
        if i2d_ig is not None and twod["fwhm"] > 0:
            if i2d_ig > 2.0 and twod["fwhm"] <= 40:
                result["graphene_layer_hint"] = "single-layer (sharp 2D, I2D/IG>2)"
            elif i2d_ig > 1.0:
                result["graphene_layer_hint"] = "few-layer"
            else:
                result["graphene_layer_hint"] = "multi-layer / graphite"
    return result


def _tmd_analysis(peaks: list[dict[str, float]]) -> dict[str, Any] | None:
    """MoS2/WS2 layer-count from E2g-A1g separation. Δ grows with layer number."""
    out: dict[str, Any] = {}
    for mat, e_rng, a_rng in (
        ("MoS2", (375, 387), (403, 412)),  # 1L ~18 cm-1 (Lee 2010)
        ("WS2", (345, 360), (413, 423)),   # A1g shift dominant indicator
    ):
        e = _find_peak_in_range(peaks, e_rng)
        a = _find_peak_in_range(peaks, a_rng)
        if not e or not a:
            continue
        delta = float(round(a["shift_cm1"] - e["shift_cm1"], 2))
        entry: dict[str, Any] = {
            "e2g_cm1": e["shift_cm1"],
            "a1g_cm1": a["shift_cm1"],
            "separation_cm1": delta,
        }
        if mat == "MoS2":
            # Lee 2010: ~18 (1L), ~21 (2L), ~25 (bulk)
            if delta <= 19.5:
                entry["layer_hint"] = "monolayer (Δ≈18 cm⁻¹)"
            elif delta <= 23:
                entry["layer_hint"] = "bilayer/few-layer"
            else:
                entry["layer_hint"] = "bulk"
        out[mat] = entry
    return out or None


def _assign_bands(peaks: list[dict[str, float]]) -> list[dict[str, Any]]:
    """Annotate peaks against the curated band table (no inference beyond table)."""
    matches = []
    for band in RAMAN_BANDS:
        lo, hi = band["range"]
        matched = [p["shift_cm1"] for p in peaks if lo <= p["shift_cm1"] <= hi]
        if matched:
            matches.append({
                "name": band["name"],
                "material": band["material"],
                "note": band["note"],
                "ref": band["ref"],
                "range_cm1": [lo, hi],
                "matched_peaks_cm1": matched,
            })
    return matches


def parse_raman(raw_text: str, laser_wavelength: float | None = None) -> dict[str, Any]:
    x, y = _parse_two_column(raw_text)
    peaks = _detect_peaks(x, y)
    return {
        "spectrum_type": "raman",
        "peaks": peaks,
        "spectrum_curve": downsample_curve(x, y, target_points=500),
        "quick_stats": {
            "rowCount": len(x),
            "xRange": [float(round(x.min(), 1)), float(round(x.max(), 1))],
            "yRange": [float(round(y.min(), 2)), float(round(y.max(), 2))],
            "peakCount": len(peaks),
        },
        "laser_wavelength_nm": laser_wavelength,
        "carbon_analysis": _carbon_analysis(peaks, x, y, laser_wavelength),
        "tmd_analysis": _tmd_analysis(peaks),
        "band_assignments": _assign_bands(peaks),
        "x_unit": "cm-1",
        "y_unit": "Intensity (a.u.)",
    }
