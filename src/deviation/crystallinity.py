"""
Crystallinity classifier with adaptive tolerance scaling.

Distinguishes bulk crystalline / nanocrystalline / amorphous samples
using 4 independent signals. Provides particle size estimate (PCM model
or Scherrer formula) when applicable.

References (verified DOIs):
  - Bersani, D. et al. (1998). Phonon confinement effects in TiO2.
    Phys. Rev. B 63, 125415. DOI: 10.1103/PhysRevB.63.125415
  - Scherrer, P. (1918). DOI: original — used Williamson 1953 modification
  - Khorsand Zak, A. et al. (2014). DOI: 10.1016/j.solidstatesciences.2014.04.012
  - Tuinstra & Koenig (1970). DOI: 10.1063/1.1674108 (amorphization)

@phase R185-5-crystallinity-classifier
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal


Classification = Literal["bulk", "nanocrystalline", "amorphous", "mixed", "unknown"]
SizeMethod = Literal["scherrer", "phonon-confinement", "qualitative"]


@dataclass
class CitationRef:
    """Lightweight citation for crystallinity outputs."""
    doi: str
    journal: str
    year: int
    title: str
    verified: bool = True


@dataclass
class CrystallinitySignals:
    """Raw signal values used for classification."""
    fwhm_ratio: float | None = None        # mean(sample_fwhm / ref_fwhm)
    peak_count_ratio: float | None = None  # matched_count / ref_count
    background_ratio: float | None = None  # background / max_peak intensity
    mean_signed_shift: float | None = None # mean(sample_pos - ref_pos)
    fwhm_cv: float | None = None           # coefficient of variation of FWHMs


@dataclass
class SizeEstimate:
    """Particle size estimate with method + uncertainty."""
    value_nm: float
    uncertainty_nm: float
    method: SizeMethod
    citation: CitationRef | None = None
    notes: str = ""


@dataclass
class CrystallinityResult:
    """Output of crystallinity classification."""
    classification: Classification
    confidence: float
    signals: CrystallinitySignals = field(default_factory=CrystallinitySignals)
    size_estimate: SizeEstimate | None = None
    tolerance_factor: float = 1.0
    reasoning: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Citations ─────────────────────────────────────────────────────────────────

CIT_BERSANI = CitationRef(
    doi="10.1103/PhysRevB.63.125415",
    journal="Physical Review B",
    year=1998,
    title="Phonon confinement effects in TiO2 nanoparticles",
)

CIT_KHORSAND = CitationRef(
    doi="10.1016/j.solidstatesciences.2014.04.012",
    journal="Solid State Sciences",
    year=2014,
    title="Williamson-Hall analysis in estimation of lattice strain",
)


# ── Signal computation helpers ────────────────────────────────────────────────

def _mean_fwhm_ratio(
    sample_peaks: list[dict[str, Any]],
    ref_peaks: list[dict[str, Any]],
    ref_fwhms_by_position: dict[float, float] | None = None,
) -> float | None:
    """
    Compute mean FWHM ratio (sample / ref) over peaks that have matching positions.

    If ref_fwhms_by_position not provided, fall back to using ref_peak["fwhm"]
    when available, else None.
    """
    if not sample_peaks:
        return None

    ratios = []
    for sp in sample_peaks:
        s_fwhm = sp.get("fwhm")
        if not s_fwhm or s_fwhm <= 0:
            continue

        # Try to find nearest ref peak
        s_pos = sp.get("shift_cm1") or sp.get("two_theta") or sp.get("shift")
        if s_pos is None:
            continue

        nearest_ref_fwhm: float | None = None
        if ref_fwhms_by_position:
            # Find closest position within tolerance
            for r_pos, r_fwhm in ref_fwhms_by_position.items():
                if abs(s_pos - r_pos) < 10 and r_fwhm > 0:
                    nearest_ref_fwhm = r_fwhm
                    break
        else:
            for rp in ref_peaks:
                r_pos = rp.get("shift") or rp.get("twotheta") or rp.get("shift_cm1")
                r_fwhm = rp.get("fwhm")
                if r_pos is None or r_fwhm is None or r_fwhm <= 0:
                    continue
                if abs(s_pos - r_pos) < 10:
                    nearest_ref_fwhm = r_fwhm
                    break

        if nearest_ref_fwhm:
            ratios.append(s_fwhm / nearest_ref_fwhm)

    return round(sum(ratios) / len(ratios), 3) if ratios else None


def _fwhm_cv(sample_peaks: list[dict[str, Any]]) -> float | None:
    """Coefficient of variation of FWHMs (std/mean). Indicates uniformity."""
    fwhms = [p["fwhm"] for p in sample_peaks if p.get("fwhm") and p["fwhm"] > 0]
    if len(fwhms) < 2:
        return None
    mean = sum(fwhms) / len(fwhms)
    if mean <= 0:
        return None
    var = sum((f - mean) ** 2 for f in fwhms) / len(fwhms)
    return round((var ** 0.5) / mean, 3)


def _background_ratio(parsed: dict[str, Any]) -> float | None:
    """
    Background / max peak intensity ratio.

    XRD parser already provides background_estimate in quality metrics.
    For Raman, estimate from min(y) / max(y) when curve is available.
    """
    # XRD path
    if "metrics" in parsed and isinstance(parsed["metrics"], dict):
        bg = parsed["metrics"].get("background_estimate")
        if bg is not None:
            peaks = parsed.get("peaks", [])
            if peaks:
                max_int = max((p.get("intensity", 0) for p in peaks), default=0)
                if max_int > 0:
                    return round(bg / max_int, 3)

    # Raman path: estimate from curve baseline
    curve = parsed.get("curve") or parsed.get("spectrum_curve")
    if curve and isinstance(curve, dict):
        y_values = curve.get("y") or curve.get("intensity")
        if y_values and len(y_values) > 10:
            sorted_y = sorted(y_values)
            # Median of lowest 10% as background estimate
            n_low = max(1, len(sorted_y) // 10)
            bg = sum(sorted_y[:n_low]) / n_low
            peak_max = sorted_y[-1]
            if peak_max > 0:
                return round(bg / peak_max, 3)

    return None


def _mean_signed_shift(matches: list[dict[str, Any]]) -> float | None:
    """Mean signed deviation across matched peaks."""
    if not matches:
        return None
    shifts = [m.get("deviation", 0) for m in matches]
    return round(sum(shifts) / len(shifts), 3) if shifts else None


# ── Size estimation ──────────────────────────────────────────────────────────

def _pcm_size_estimate(fwhm_ratio: float, mean_shift_cm1: float | None = None) -> SizeEstimate:
    """
    Phonon confinement model (PCM) particle size estimate.

    Approximation for TiO2 anatase Eg mode and similar oxides:
      D (nm) ~ 10 / (fwhm_ratio - 0.5)

    Coefficient varies by material — this is a ballpark estimate, not refinement.
    Bersani 1998 (DOI: 10.1103/PhysRevB.63.125415) for TiO2; values reasonable
    for many oxide nanoparticles within factor of 2.
    """
    if fwhm_ratio <= 0.5:
        size = 50.0  # ratio too small for confinement
        uncertainty = 20.0
    else:
        size = round(10 / (fwhm_ratio - 0.5), 1)
        # Uncertainty grows with size (PCM less reliable for large particles)
        uncertainty = round(size * 0.4, 1)

    return SizeEstimate(
        value_nm=size,
        uncertainty_nm=uncertainty,
        method="phonon-confinement",
        citation=CIT_BERSANI,
        notes=(
            "PCM rough estimate. ±40% uncertainty. "
            "Confirm via TEM/SEM particle size distribution."
        ),
    )


def _scherrer_size_estimate(
    fwhm_deg: float,
    two_theta_deg: float,
    wavelength_nm: float = 0.15406,  # Cu Kalpha1
    K: float = 0.9,
) -> SizeEstimate:
    """
    Scherrer formula: D = K * lambda / (beta * cos(theta))
    Returns crystallite size estimate.
    """
    import math
    if fwhm_deg <= 0 or two_theta_deg <= 0:
        return SizeEstimate(0, 0, "scherrer", citation=CIT_KHORSAND,
                            notes="Invalid FWHM/2theta input")

    beta_rad = math.radians(fwhm_deg)
    theta_rad = math.radians(two_theta_deg / 2)
    cos_theta = math.cos(theta_rad)

    if cos_theta <= 0:
        return SizeEstimate(0, 0, "scherrer", citation=CIT_KHORSAND,
                            notes="Invalid theta")

    size = K * wavelength_nm / (beta_rad * cos_theta)
    return SizeEstimate(
        value_nm=round(size, 2),
        uncertainty_nm=round(size * 0.15, 2),  # ~15% typical Scherrer uncertainty
        method="scherrer",
        citation=CIT_KHORSAND,
        notes="Scherrer formula with K=0.9 (spherical crystallites assumed).",
    )


# ── Main classifier ──────────────────────────────────────────────────────────

def classify_crystallinity(
    spectrum_type: str,
    parsed: dict[str, Any],
    sample_peaks: list[dict[str, Any]] | None = None,
    ref_peaks: list[dict[str, Any]] | None = None,
    matches: list[dict[str, Any]] | None = None,
) -> CrystallinityResult:
    """
    Classify sample crystallinity from spectrum + match data.

    Args:
        spectrum_type: raman/xrd/ftir/pl/uvvis
        parsed: parser output (may include "metrics" and "curve")
        sample_peaks: list of detected peaks (for FWHM, intensity)
        ref_peaks: reference peaks from materialProfile
        matches: match dicts (PeakMatch.to_dict()) for shift analysis

    Returns:
        CrystallinityResult with classification + confidence + size + tolerance factor.
    """
    sample_peaks = sample_peaks or parsed.get("peaks", [])
    ref_peaks = ref_peaks or []
    matches = matches or []

    signals = CrystallinitySignals()
    reasoning: list[str] = []

    # ── Signal 1: FWHM ratio ─────────────────────────────────────────────────
    if ref_peaks:
        ratio = _mean_fwhm_ratio(sample_peaks, ref_peaks)
        signals.fwhm_ratio = ratio
        if ratio:
            reasoning.append(f"Mean FWHM ratio vs ref: {ratio:.2f}")
    elif sample_peaks:
        # No ref → can still report mean FWHM CV
        signals.fwhm_cv = _fwhm_cv(sample_peaks)

    # ── Signal 2: peak count ratio ───────────────────────────────────────────
    if ref_peaks and matches:
        pcr = len(matches) / len(ref_peaks) if ref_peaks else 0
        signals.peak_count_ratio = round(pcr, 3)
        reasoning.append(f"Detected {len(matches)}/{len(ref_peaks)} ref peaks ({pcr*100:.0f}%)")

    # ── Signal 3: background ratio ───────────────────────────────────────────
    bg_ratio = _background_ratio(parsed)
    if bg_ratio is not None:
        signals.background_ratio = bg_ratio
        reasoning.append(f"Background/peak ratio: {bg_ratio:.2f}")

    # ── Signal 4: mean signed shift ──────────────────────────────────────────
    if matches:
        signals.mean_signed_shift = _mean_signed_shift(matches)

    # ── FWHM CV (uniform vs non-uniform broadening) ─────────────────────────
    if signals.fwhm_cv is None and sample_peaks:
        signals.fwhm_cv = _fwhm_cv(sample_peaks)

    # ── Classification logic ────────────────────────────────────────────────
    classification: Classification = "unknown"
    confidence = 0.0
    size_est: SizeEstimate | None = None
    tolerance_factor = 1.0

    fwhm_r = signals.fwhm_ratio
    pcr = signals.peak_count_ratio
    bg = signals.background_ratio

    if fwhm_r is not None:
        # Strong signals from FWHM ratio
        if fwhm_r < 1.3:
            classification = "bulk"
            confidence = 0.85
            tolerance_factor = 1.0
            reasoning.append("FWHM near reference -> bulk crystalline")
        elif fwhm_r < 2.5:
            classification = "nanocrystalline"
            confidence = 0.75
            tolerance_factor = 1.5
            reasoning.append("FWHM broadened 1.3-2.5x -> nanocrystalline (PCM regime)")
            # Try size estimate
            if spectrum_type in ("raman", "ftir"):
                size_est = _pcm_size_estimate(fwhm_r, signals.mean_signed_shift)
        elif fwhm_r < 4.0:
            classification = "nanocrystalline"
            confidence = 0.80
            tolerance_factor = 2.0
            reasoning.append("FWHM broadened 2.5-4x -> small nanocrystals (<5 nm)")
            if spectrum_type in ("raman", "ftir"):
                size_est = _pcm_size_estimate(fwhm_r, signals.mean_signed_shift)
        else:
            classification = "amorphous"
            confidence = 0.75
            tolerance_factor = 3.0
            reasoning.append(f"FWHM broadened {fwhm_r:.1f}x -> amorphous tendency")
    else:
        # No ref → use absolute signals
        if pcr is not None and pcr < 0.4:
            classification = "amorphous"
            confidence = 0.6
            tolerance_factor = 3.0
            reasoning.append("Too few ref peaks matched -> amorphous-leaning")
        else:
            reasoning.append("Insufficient signals to classify (no ref FWHM data)")

    # Adjust confidence using background ratio (corroborating evidence)
    if bg is not None:
        if classification == "amorphous" and bg > 0.3:
            confidence = min(0.95, confidence + 0.15)
            reasoning.append(f"High background ({bg:.2f}) corroborates amorphous")
        elif classification == "bulk" and bg < 0.1:
            confidence = min(0.95, confidence + 0.05)
        elif classification == "nanocrystalline" and 0.1 < bg < 0.3:
            confidence = min(0.95, confidence + 0.1)
            reasoning.append("Moderate background corroborates nanocrystalline")

    # PCR corroboration
    if pcr is not None:
        if classification == "bulk" and pcr < 0.6:
            confidence -= 0.2
            reasoning.append("Bulk classification weakened by low peak count")
        elif classification == "amorphous" and pcr < 0.4:
            confidence = min(0.95, confidence + 0.1)

    confidence = round(max(0.0, min(0.95, confidence)), 2)

    return CrystallinityResult(
        classification=classification,
        confidence=confidence,
        signals=signals,
        size_estimate=size_est,
        tolerance_factor=tolerance_factor,
        reasoning=reasoning,
    )


def adaptive_tolerance(base_tolerance: float, classification: Classification) -> float:
    """Scale matcher tolerance based on crystallinity classification."""
    factors = {
        "bulk": 1.0,
        "nanocrystalline": 1.5,
        "amorphous": 3.0,
        "mixed": 2.0,
        "unknown": 1.0,
    }
    return base_tolerance * factors.get(classification, 1.0)
