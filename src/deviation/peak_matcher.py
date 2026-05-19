"""
Deterministic peak matcher using Hungarian algorithm.

Matches a list of sample peaks to a list of reference peaks based on
position proximity (cm-1 for Raman/FTIR, degrees for XRD, eV for PL,
nm for UV-Vis). Returns one-to-one assignment minimizing total deviation.

Algorithm:
  1. Build cost matrix C[i,j] = |position_sample[i] - position_ref[j]|
  2. Cap cost at `tolerance` for peaks too far apart
  3. Solve assignment problem (scipy.optimize.linear_sum_assignment)
  4. Filter assignments exceeding tolerance → unmatched

@phase R185-1-deterministic-peak-matcher
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

import numpy as np
from scipy.optimize import linear_sum_assignment


# ── Tolerances per spectrum type ──────────────────────────────────────────────
# Tuned to typical experimental + instrumental precision.
# References:
#   Raman ±5 cm-1: typical spectrometer resolution + thermal broadening
#   XRD ±0.3 deg: typical Cu Kalpha line + sample displacement error
#   FTIR ±10 cm-1: lower resolution + functional group flexibility
#   PL ±0.05 eV: exciton broadening at RT
#   UV-Vis ±10 nm: absorption edge fuzziness

SpectrumType = Literal["raman", "xrd", "ftir", "pl", "uvvis"]

DEFAULT_TOLERANCES: dict[SpectrumType, float] = {
    "raman": 5.0,    # cm-1
    "xrd": 0.3,      # degrees (2-theta)
    "ftir": 10.0,    # cm-1
    "pl": 0.05,      # eV
    "uvvis": 10.0,   # nm
}


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class PeakMatch:
    """A single peak assignment between sample and reference."""
    sample_index: int
    sample_position: float
    sample_intensity: float
    sample_fwhm: float | None
    ref_index: int
    ref_position: float
    ref_intensity: float
    ref_assignment: str
    deviation: float  # signed: sample - ref
    confidence: float  # 0.0 - 1.0


@dataclass
class UnmatchedPeak:
    """A sample peak with no good ref match (and vice versa)."""
    side: Literal["sample", "ref"]
    index: int
    position: float
    intensity: float
    note: str = ""


@dataclass
class MatchResult:
    """Output of peak matching."""
    spectrum_type: str
    reference_formula: str
    reference_label: str  # e.g. "MoS2 (2H)" or "WO3 (monoclinic)"
    tolerance_used: float
    matches: list[PeakMatch] = field(default_factory=list)
    unmatched_sample: list[UnmatchedPeak] = field(default_factory=list)
    unmatched_ref: list[UnmatchedPeak] = field(default_factory=list)

    # Aggregate metrics
    match_count: int = 0
    match_rate: float = 0.0  # matches / total_sample_peaks
    mean_abs_deviation: float = 0.0
    max_abs_deviation: float = 0.0
    rmse: float = 0.0
    quality_grade: Literal["excellent", "good", "fair", "poor"] = "poor"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Position extractor ────────────────────────────────────────────────────────

def _get_position(peak: dict[str, Any], spectrum_type: str) -> float | None:
    """Extract position from peak dict based on spectrum type.

    Sample peak keys (from parsers): shift_cm1, two_theta, energy_ev, wavelength_nm
    Reference peak keys (from materialProfiles): shift, twotheta, energy, wavelength
    """
    if spectrum_type in ("raman", "ftir"):
        return peak.get("shift_cm1") or peak.get("shift")
    if spectrum_type == "xrd":
        return peak.get("two_theta") or peak.get("twotheta") or peak.get("twoTheta")
    if spectrum_type == "pl":
        return peak.get("energy_ev") or peak.get("energy")
    if spectrum_type == "uvvis":
        return peak.get("wavelength_nm") or peak.get("wavelength")
    return None


def _get_intensity(peak: dict[str, Any]) -> float:
    """Extract intensity, with fallbacks."""
    for key in ("relative_intensity", "intensity"):
        if key in peak and peak[key] is not None:
            return float(peak[key])
    return 0.0


# ── Confidence + quality grading ──────────────────────────────────────────────

def _confidence(deviation: float, tolerance: float, ref_intensity: float) -> float:
    """Confidence score combining position accuracy and ref-peak prominence.

    - Position factor: 1.0 at zero deviation, 0.0 at tolerance threshold
    - Intensity factor: weight by ref intensity (strong peaks more reliable)
    """
    pos_factor = max(0.0, 1.0 - abs(deviation) / tolerance)
    int_factor = min(1.0, ref_intensity / 100.0)  # ref intensity is 0-100
    # 70/30 weight: position dominates
    return round(0.7 * pos_factor + 0.3 * int_factor, 3)


def _grade_match(match_rate: float, mean_abs_dev: float, tolerance: float) -> str:
    """Overall match quality grade."""
    rel_dev = mean_abs_dev / tolerance if tolerance > 0 else 1.0
    if match_rate >= 0.8 and rel_dev < 0.3:
        return "excellent"
    if match_rate >= 0.6 and rel_dev < 0.5:
        return "good"
    if match_rate >= 0.4 and rel_dev < 0.8:
        return "fair"
    return "poor"


# ── Main matcher ──────────────────────────────────────────────────────────────

def match_peaks(
    sample_peaks: list[dict[str, Any]],
    ref_peaks: list[dict[str, Any]],
    spectrum_type: SpectrumType,
    reference_formula: str,
    reference_label: str = "",
    tolerance: float | None = None,
) -> MatchResult:
    """
    Match sample peaks to reference peaks using Hungarian algorithm.

    Args:
        sample_peaks: list of parsed peak dicts from spectra parser
        ref_peaks: list of reference peaks from materialProfiles
        spectrum_type: one of "raman" | "xrd" | "ftir" | "pl" | "uvvis"
        reference_formula: e.g. "MoS2"
        reference_label: human-readable, e.g. "MoS2 (2H, monolayer)"
        tolerance: position tolerance. If None, use DEFAULT_TOLERANCES.

    Returns:
        MatchResult with per-peak assignments + aggregate metrics.
    """
    tol = tolerance if tolerance is not None else DEFAULT_TOLERANCES[spectrum_type]

    result = MatchResult(
        spectrum_type=spectrum_type,
        reference_formula=reference_formula,
        reference_label=reference_label or reference_formula,
        tolerance_used=tol,
    )

    if not sample_peaks or not ref_peaks:
        return result

    # Extract positions
    sample_pos = []
    for i, p in enumerate(sample_peaks):
        pos = _get_position(p, spectrum_type)
        if pos is not None:
            sample_pos.append((i, pos, _get_intensity(p), p.get("fwhm")))

    ref_pos = []
    for j, p in enumerate(ref_peaks):
        pos = _get_position(p, spectrum_type)
        if pos is not None:
            ref_pos.append((j, pos, _get_intensity(p), p.get("assignment", "")))

    if not sample_pos or not ref_pos:
        return result

    # Build cost matrix (capped at 10*tolerance for unmatched cases)
    n_s = len(sample_pos)
    n_r = len(ref_pos)
    cost_cap = 10 * tol
    cost = np.full((n_s, n_r), cost_cap, dtype=float)

    for i, (_, s_pos, _, _) in enumerate(sample_pos):
        for j, (_, r_pos, _, _) in enumerate(ref_pos):
            d = abs(s_pos - r_pos)
            cost[i, j] = min(d, cost_cap)

    # Solve assignment
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_sample = set()
    matched_ref = set()
    deviations = []

    for i, j in zip(row_ind, col_ind):
        d = cost[i, j]
        if d > tol:
            continue  # exceeds tolerance, leave both unmatched

        s_idx, s_pos, s_int, s_fwhm = sample_pos[i]
        r_idx, r_pos, r_int, r_assign = ref_pos[j]
        signed_dev = round(s_pos - r_pos, 4)

        result.matches.append(PeakMatch(
            sample_index=s_idx,
            sample_position=s_pos,
            sample_intensity=s_int,
            sample_fwhm=s_fwhm,
            ref_index=r_idx,
            ref_position=r_pos,
            ref_intensity=r_int,
            ref_assignment=r_assign,
            deviation=signed_dev,
            confidence=_confidence(signed_dev, tol, r_int),
        ))
        matched_sample.add(i)
        matched_ref.add(j)
        deviations.append(abs(signed_dev))

    # Collect unmatched
    for i, (s_idx, s_pos, s_int, _) in enumerate(sample_pos):
        if i not in matched_sample:
            result.unmatched_sample.append(UnmatchedPeak(
                side="sample", index=s_idx, position=s_pos, intensity=s_int,
                note="No reference peak within tolerance",
            ))

    for j, (r_idx, r_pos, r_int, r_assign) in enumerate(ref_pos):
        if j not in matched_ref:
            result.unmatched_ref.append(UnmatchedPeak(
                side="ref", index=r_idx, position=r_pos, intensity=r_int,
                note=f"Expected ref peak not observed: {r_assign}",
            ))

    # Aggregate metrics
    result.match_count = len(result.matches)
    result.match_rate = round(result.match_count / n_s, 3) if n_s > 0 else 0.0
    if deviations:
        result.mean_abs_deviation = round(float(np.mean(deviations)), 4)
        result.max_abs_deviation = round(float(np.max(deviations)), 4)
        result.rmse = round(float(np.sqrt(np.mean(np.array(deviations) ** 2))), 4)
    result.quality_grade = _grade_match(
        result.match_rate, result.mean_abs_deviation, tol,
    )

    return result
