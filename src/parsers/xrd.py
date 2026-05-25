"""XRD parser with Williamson-Hall + Scherrer + spectrum curve."""

from __future__ import annotations

import logging
import math
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter

from src.parsers._tabular import parse_csv_two_column, parse_xlsx_two_column
from src.parsers._utils import downsample_curve

logger = logging.getLogger(__name__)

# Cu K-alpha1 wavelength in Angstroms (most common XRD source)
CU_KA1_ANGSTROM = 1.5406

# Profile function types (R161-phase-E)
PROFILE_GAUSSIAN = "gaussian"
PROFILE_LORENTZIAN = "lorentzian"
PROFILE_PSEUDOVOIGT = "pseudo_voigt"
DEFAULT_PROFILE = PROFILE_PSEUDOVOIGT  # Most realistic for typical lab XRD


def _gaussian(x, amp, center, fwhm):
    sigma = fwhm / (2 * math.sqrt(2 * math.log(2)))
    return amp * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


def _lorentzian(x, amp, center, fwhm):
    gamma = fwhm / 2
    return amp * (gamma ** 2) / ((x - center) ** 2 + gamma ** 2)


def _pseudo_voigt(x, amp, center, fwhm, eta):
    # eta: 0 = pure Gaussian, 1 = pure Lorentzian
    return eta * _lorentzian(x, amp, center, fwhm) + (1 - eta) * _gaussian(x, amp, center, fwhm)


def _fit_peak_profile(
    x: np.ndarray,
    y: np.ndarray,
    peak_idx: int,
    initial_fwhm: float,
    profile: str = DEFAULT_PROFILE,
    window_pts: int = 15,
) -> dict[str, float] | None:
    """Fit selected profile function to peak around peak_idx.

    Returns {fwhm, eta, amp, center, r_squared} or None on failure.
    """
    try:
        # Extract window around peak
        i_start = max(0, peak_idx - window_pts)
        i_end = min(len(x), peak_idx + window_pts + 1)
        x_win = x[i_start:i_end]
        y_win = y[i_start:i_end]
        # Subtract local baseline
        baseline = float(np.min(y_win))
        y_fit = y_win - baseline
        amp_init = float(y_fit.max())
        center_init = float(x[peak_idx])

        if profile == PROFILE_GAUSSIAN:
            p0 = [amp_init, center_init, initial_fwhm]
            popt, _ = curve_fit(_gaussian, x_win, y_fit, p0=p0, maxfev=2000)
            y_pred = _gaussian(x_win, *popt)
            result = {"amp": popt[0], "center": popt[1], "fwhm": popt[2], "eta": 0.0}
        elif profile == PROFILE_LORENTZIAN:
            p0 = [amp_init, center_init, initial_fwhm]
            popt, _ = curve_fit(_lorentzian, x_win, y_fit, p0=p0, maxfev=2000)
            y_pred = _lorentzian(x_win, *popt)
            result = {"amp": popt[0], "center": popt[1], "fwhm": popt[2], "eta": 1.0}
        else:  # pseudo_voigt
            p0 = [amp_init, center_init, initial_fwhm, 0.5]
            popt, _ = curve_fit(
                _pseudo_voigt, x_win, y_fit, p0=p0,
                bounds=([0, x_win[0], 0, 0], [np.inf, x_win[-1], 5, 1]),
                maxfev=2000,
            )
            y_pred = _pseudo_voigt(x_win, *popt)
            result = {"amp": popt[0], "center": popt[1], "fwhm": popt[2], "eta": popt[3]}

        # R² goodness of fit
        ss_res = float(np.sum((y_fit - y_pred) ** 2))
        ss_tot = float(np.sum((y_fit - y_fit.mean()) ** 2))
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        result["r_squared"] = max(0.0, r_squared)
        return result
    except Exception:  # noqa: BLE001
        return None


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
                # B3: accept grazing-incidence/GISAXS (low-angle) + high-angle-only
                # scans. Only reject clearly-out-of-range or non-monotone data.
                if 0.1 < x.min() and x.max() <= 180 and x.max() > x.min():
                    return x, y
        except Exception:  # noqa: BLE001
            continue
    raise ValueError("Could not parse two-column XRD data")


def _detect_peaks(
    x: np.ndarray, y: np.ndarray, *,
    max_peaks: int = 30,
    profile: str = DEFAULT_PROFILE,
) -> list[dict[str, float]]:
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
    prominences = props.get("prominences", np.zeros(len(peak_idx)))
    y_max = y_smooth.max()
    for i, idx in enumerate(peak_idx):
        dx = float(x[1] - x[0]) if len(x) > 1 else 0.0
        rough_fwhm = float(widths[i]) * dx
        two_theta = float(x[idx])
        intensity = float(y_smooth[idx])

        # Profile function fit (R161-phase-E)
        fit = _fit_peak_profile(x, y_smooth, idx, rough_fwhm, profile=profile)
        if fit and fit.get("r_squared", 0) > 0.5:
            fwhm = fit["fwhm"]
            two_theta = fit["center"]  # refined position
            profile_eta = fit["eta"]
            fit_r2 = fit["r_squared"]
        else:
            fwhm = rough_fwhm
            profile_eta = None
            fit_r2 = None

        # Integral breadth: depends on profile shape
        # Gaussian: β = FWHM · sqrt(π/(4ln2)) ≈ 1.0645·FWHM
        # Lorentzian: β = FWHM · π/2 ≈ 1.5708·FWHM
        # Pseudo-Voigt: weighted by eta
        if profile_eta is not None:
            beta = fwhm * (profile_eta * 1.5708 + (1 - profile_eta) * 1.0645)
        else:
            beta = fwhm * 1.0645  # default Gaussian

        peaks.append({
            "two_theta": float(round(two_theta, 3)),
            "intensity": float(round(intensity, 2)),
            "fwhm": float(round(fwhm, 4)),
            "integral_breadth": float(round(beta, 4)),
            "relative_intensity": float(round(intensity / y_max * 100, 1)),
            "prominence": float(round(float(prominences[i]), 2)),
            "profile_eta": float(round(profile_eta, 3)) if profile_eta is not None else None,
            "fit_r_squared": float(round(fit_r2, 3)) if fit_r2 is not None else None,
        })
    return peaks


def _enrich_peaks(
    peaks: list[dict[str, float]],
    wavelength: float = CU_KA1_ANGSTROM,
) -> list[dict[str, float]]:
    """Add per-peak derived properties: d-spacing, Scherrer D, dislocation density, microstrain.

    Returns NEW list with enriched dicts.
    """
    enriched = []
    for p in peaks:
        theta_rad = math.radians(p["two_theta"] / 2.0)
        sin_theta = math.sin(theta_rad)
        cos_theta = math.cos(theta_rad)

        # d-spacing (Bragg's law): d = λ / (2·sinθ)
        d_spacing = wavelength / (2 * sin_theta) if sin_theta > 0 else None

        # Per-peak Scherrer crystallite size D (nm)
        fwhm_rad = math.radians(p["fwhm"])
        D_nm = None
        # B1: guard cos_theta (→0 as 2θ→180°) to avoid inf/negative crystallite size
        if fwhm_rad > 0 and cos_theta > 1e-6:
            D_angstrom = (K_SCHERRER * wavelength) / (fwhm_rad * cos_theta)
            D_nm = D_angstrom / 10.0

        # Dislocation density δ = 1/D² (lines/m²)
        dislocation_density = None
        if D_nm and D_nm > 0:
            D_m = D_nm * 1e-9  # nm → m
            dislocation_density = 1.0 / (D_m * D_m)

        # Microstrain ε per-peak: β·cosθ / 4
        microstrain = None
        if fwhm_rad > 0:
            microstrain = (fwhm_rad * cos_theta) / 4.0

        enriched.append({
            **p,
            "d_spacing": float(round(d_spacing, 4)) if d_spacing else None,  # Å
            "crystallite_size_nm": float(round(D_nm, 2)) if D_nm else None,
            "dislocation_density": float(round(dislocation_density, 3)) if dislocation_density else None,
            "microstrain": float(round(microstrain, 6)) if microstrain else None,
        })
    return enriched


def _crystallinity_percent(x: np.ndarray, y: np.ndarray, peaks: list[dict]) -> float | None:
    """Estimate crystallinity %: (sum of peak areas) / (total area).

    Peak area approximated by I × FWHM (Gaussian: 1.0645·I·FWHM).
    Total area = trapezoidal integral.
    """
    if not peaks or len(y) < 10:
        return None
    total_area = float(np.trapezoid(y, x))
    if total_area <= 0:
        return None
    crystalline_area = sum(p["intensity"] * p["fwhm"] * 1.0645 for p in peaks)
    return float(round(min(100.0, crystalline_area / total_area * 100), 1))


def _quality_metrics(x: np.ndarray, y: np.ndarray, peaks: list[dict]) -> dict:
    """Scan quality + data acquisition metadata."""
    metrics = {
        "scan_range_2theta": [float(round(x[0], 3)), float(round(x[-1], 3))],
        "step_size_deg": float(round((x[-1] - x[0]) / (len(x) - 1), 4)) if len(x) > 1 else 0,
        "data_points": int(len(x)),
        "n_peaks_detected": len(peaks),
    }
    if peaks:
        intensities = np.array([p["intensity"] for p in peaks])
        bg = float(np.percentile(y, 10))  # low percentile as bg estimate
        noise_std = float(np.std(y[y < np.percentile(y, 30)]))
        max_intensity = float(intensities.max())
        metrics["background_estimate"] = float(round(bg, 2))
        metrics["noise_std"] = float(round(noise_std, 2))
        metrics["snr"] = float(round(max_intensity / noise_std, 1)) if noise_std > 0 else None
        metrics["max_intensity"] = float(round(max_intensity, 2))
        metrics["smallest_fwhm"] = float(round(min(p["fwhm"] for p in peaks), 4))
        metrics["resolution_estimate"] = metrics["smallest_fwhm"]
    return metrics


def _scherrer_crystallite_size(peaks: list[dict[str, float]], wavelength: float = CU_KA1_ANGSTROM) -> float | None:
    """Average Scherrer crystallite size (nm) from top 3 peaks."""
    if len(peaks) < 3:
        return None
    top3 = sorted(peaks, key=lambda p: -p["relative_intensity"])[:3]
    sizes = []
    for p in top3:
        theta_rad = math.radians(p["two_theta"] / 2.0)
        fwhm_rad = math.radians(p["fwhm"])
        # B2: guard fwhm AND cos_theta (→0 at high angle) before division
        if fwhm_rad <= 0 or math.cos(theta_rad) <= 1e-6:
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


def parse_xrd(
    raw_text: str, *,
    wavelength: float = CU_KA1_ANGSTROM,
    anode: str | None = None,
    monochromator: str | None = None,
    zero_shift: float = 0.0,
    profile: str = DEFAULT_PROFILE,
) -> dict[str, Any]:
    x, y = _parse_two_column(raw_text)
    if anode:
        wavelength = resolve_effective_wavelength(anode, monochromator)
    # Apply zero shift correction (R161-phase-E)
    if zero_shift != 0.0:
        x = x - zero_shift
    return _build_xrd_result(x, y, wavelength=wavelength, profile=profile)


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
    zero_shift: float = 0.0,
    profile: str = DEFAULT_PROFILE,
) -> dict:
    """Parse XRD + attach citation candidates if formula resolvable."""
    from src.citation.lookup import lookup_xrd_candidates

    parsed = parse_xrd(raw_text, anode=anode, monochromator=monochromator, zero_shift=zero_shift, profile=profile)
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
def parse_xrd_bytes(
    raw_bytes: bytes,
    filename: str,
    anode: str | None = None,
    monochromator: str | None = None,
    zero_shift: float = 0.0,
    profile: str = DEFAULT_PROFILE,
) -> dict[str, Any]:
    """Parse XRD from raw bytes. Routes to .xlsx parser if .xlsx, else text."""
    if filename.lower().endswith(".xlsx"):
        result = parse_xlsx_two_column(raw_bytes)
        if result is None:
            raise ValueError("Could not parse XLSX (no valid 2theta+intensity columns)")
        x, y = result
        wavelength = resolve_effective_wavelength(anode, monochromator)
        if zero_shift != 0.0:
            x = x - zero_shift
        return _build_xrd_result(x, y, wavelength=wavelength, profile=profile)
    # Decode bytes as text
    try:
        text_str = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text_str = raw_bytes.decode("latin-1", errors="replace")
    return parse_xrd(text_str, anode=anode, monochromator=monochromator, zero_shift=zero_shift, profile=profile)


def _build_xrd_result(
    x: np.ndarray, y: np.ndarray, *,
    wavelength: float = CU_KA1_ANGSTROM,
    profile: str = DEFAULT_PROFILE,
) -> dict[str, Any]:
    """Build XRD result dict from x,y arrays (shared with parse_xrd)."""
    raw_peaks = _detect_peaks(x, y, profile=profile)
    peaks = _enrich_peaks(raw_peaks, wavelength) if raw_peaks else []
    scherrer = _scherrer_crystallite_size(peaks, wavelength) if peaks else None
    wh = _williamson_hall(peaks, wavelength) if peaks else None
    crystallinity = _crystallinity_percent(x, y, peaks) if peaks else None
    quality = _quality_metrics(x, y, peaks)
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
        "crystallinity_percent": crystallinity,
        "quality_metrics": quality,
        "wavelength_angstrom": wavelength,
        "source": next((k for k, v in ANODE_WAVELENGTHS.items() if abs(v - wavelength) < 0.001), "Cu") + "-Kα1",
    }
