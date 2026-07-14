"""
Multi-phase matcher: greedy iterative peak assignment.

Algorithm:
  1. Sort candidate phases by weight prior (user-declared role + fraction)
  2. For each phase in order: match peaks against its reference, remove
     matched sample peaks from the working pool
  3. Continue until all phases processed
  4. Remaining peaks = unassigned (potential contamination / new phase)

Researcher intent vs sample reality:
  - User declares: Sample.composition = [(MoS2, matrix, 0.7), (C, support, 0.3)]
  - Engine outputs: per-component MatchResult + which intent peaks were observed
  - Surfaces: intended but missing / observed but not intended

Self-implemented per Option A (R185 license audit). Algorithm inspired by:
  Castelli et al., XERUS (Adv Theory Simul 2022, doi:10.1002/adts.202100588)
  Code is original; no GPL/BGMN dependency.

@phase R185-4-multi-phase-matcher
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.deviation.peak_matcher import (
    DEFAULT_TOLERANCES,
    MatchResult,
    SpectrumType,
    match_peaks,
)

# Weight priors per role — used to order phases in greedy matching.
# Higher weight = matched first → its peaks taken from pool before next phase.
ROLE_WEIGHT: dict[str, float] = {
    "matrix": 1.0,       # primary phase
    "core": 0.95,        # core-shell
    "active": 0.9,       # functional layer
    "shell": 0.8,
    "support": 0.6,      # rGO support, etc.
    "filler": 0.5,
    "dopant": 0.4,       # small fraction, distinct signatures
    "substrate": 0.3,    # baseline noise contributor
}


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class ComponentDeclaration:
    """User-declared composition entry from Sample.composition."""
    formula: str
    role: str = "matrix"
    nominal_fraction: float | None = None
    formation_method: str | None = None


@dataclass
class ComponentMatch:
    """Per-component match result + intent reconciliation."""
    formula: str
    role: str
    weight_prior: float
    nominal_fraction: float | None
    reference_label: str
    match_result: dict[str, Any]  # MatchResult.to_dict()
    intended_peaks_observed: int
    intended_peaks_total: int
    intent_coverage: float  # observed / total

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MultiPhaseResult:
    """Output of multi-phase greedy matching."""
    spectrum_type: str
    components: list[ComponentMatch] = field(default_factory=list)
    unassigned_peaks: list[dict[str, Any]] = field(default_factory=list)
    intended_phases: list[str] = field(default_factory=list)
    intended_but_not_observed: list[str] = field(default_factory=list)
    overall_match_rate: float = 0.0
    overall_grade: str = "poor"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_position(peak: dict[str, Any], spectrum_type: str) -> float | None:
    """Same as peak_matcher._get_position — local copy to keep modules decoupled."""
    if spectrum_type in ("raman", "ftir"):
        return peak.get("shift_cm1") or peak.get("shift")
    if spectrum_type == "xrd":
        return peak.get("two_theta") or peak.get("twotheta") or peak.get("twoTheta")
    if spectrum_type == "pl":
        return peak.get("energy_ev") or peak.get("energy")
    if spectrum_type == "uvvis":
        return peak.get("wavelength_nm") or peak.get("wavelength")
    return None


def _resolve_weight(component: ComponentDeclaration) -> float:
    """Combine role weight + nominal fraction prior."""
    base = ROLE_WEIGHT.get(component.role, 0.5)
    if component.nominal_fraction is not None:
        # Boost by fraction — high-fraction phases match first
        return base * (0.5 + component.nominal_fraction * 0.5)
    return base


def _matched_positions(match_result: MatchResult, spectrum_type: str) -> set[float]:
    """Return set of sample positions that were matched."""
    return {round(m.sample_position, 4) for m in match_result.matches}


def _grade_overall(rate: float) -> str:
    if rate >= 0.85: return "excellent"
    if rate >= 0.7: return "good"
    if rate >= 0.5: return "fair"
    return "poor"


# ── Main multi-phase matcher ──────────────────────────────────────────────────

def match_multi_phase(
    sample_peaks: list[dict[str, Any]],
    components: list[ComponentDeclaration],
    profile_loader,  # callable: (formula: str) -> material_profile dict or None
    spectrum_type: SpectrumType,
    tolerance: float | None = None,
) -> MultiPhaseResult:
    """
    Greedy iterative multi-phase peak matching.

    Args:
        sample_peaks: parsed peaks from spectrum
        components: user-declared composition list
        profile_loader: function to fetch materialProfile for a formula
        spectrum_type: raman / xrd / ftir / pl / uvvis
        tolerance: position tolerance, defaults per spectrum_type

    Returns:
        MultiPhaseResult with per-component matches + unassigned peaks +
        intent reconciliation (intended_but_not_observed).
    """
    tol = tolerance if tolerance is not None else DEFAULT_TOLERANCES[spectrum_type]
    sig_key_map = {
        "raman": "raman", "ftir": "ftir", "xrd": "xrd",
        "pl": "pl", "uvvis": "uvvis",
    }
    sig_key = sig_key_map[spectrum_type]

    result = MultiPhaseResult(
        spectrum_type=spectrum_type,
        intended_phases=[c.formula for c in components],
    )

    # Sort components by weight prior (descending)
    ordered = sorted(
        components,
        key=lambda c: -_resolve_weight(c),
    )

    # Working pool of unassigned sample peaks (kept as dicts, indexed)
    working_pool: list[dict[str, Any]] = list(sample_peaks)
    pool_original_indices: list[int] = list(range(len(sample_peaks)))

    total_matched = 0

    for component in ordered:
        profile = profile_loader(component.formula)
        if not profile:
            result.intended_but_not_observed.append(component.formula)
            continue

        signatures = profile.get("spectralSignatures", {})
        sig = signatures.get(sig_key) if signatures else None
        if not sig or not sig.get("peaks"):
            result.intended_but_not_observed.append(component.formula)
            continue

        ref_peaks = sig["peaks"]
        ref_count = len(ref_peaks)

        label = component.formula
        common_names = profile.get("commonNames", [])
        if common_names:
            label = f"{component.formula} ({common_names[0]})"

        # Run single-phase matcher on the current working pool
        single_result = match_peaks(
            sample_peaks=working_pool,
            ref_peaks=ref_peaks,
            spectrum_type=spectrum_type,
            reference_formula=component.formula,
            reference_label=label,
            tolerance=tol,
        )

        # Intent reconciliation: how many of THIS component's ref peaks were observed?
        observed_count = single_result.match_count
        intent_coverage = round(observed_count / ref_count, 3) if ref_count else 0.0

        # If intent_coverage too low, flag as "intended but not observed"
        if intent_coverage < 0.3:
            result.intended_but_not_observed.append(component.formula)

        result.components.append(ComponentMatch(
            formula=component.formula,
            role=component.role,
            weight_prior=round(_resolve_weight(component), 3),
            nominal_fraction=component.nominal_fraction,
            reference_label=label,
            match_result=single_result.to_dict(),
            intended_peaks_observed=observed_count,
            intended_peaks_total=ref_count,
            intent_coverage=intent_coverage,
        ))

        total_matched += observed_count

        # Remove matched sample peaks from working pool (greedy)
        matched_pool_indices = {m.sample_index for m in single_result.matches}
        new_pool = []
        new_indices = []
        for i, (peak, orig_idx) in enumerate(zip(working_pool, pool_original_indices, strict=True)):
            if i not in matched_pool_indices:
                new_pool.append(peak)
                new_indices.append(orig_idx)
        working_pool = new_pool
        pool_original_indices = new_indices

    # Remaining peaks = unassigned
    for peak, orig_idx in zip(working_pool, pool_original_indices, strict=True):
        pos = _get_position(peak, spectrum_type)
        if pos is None:
            continue
        intensity = float(peak.get("relative_intensity", peak.get("intensity", 0)))
        if intensity < 5:  # filter noise
            continue
        result.unassigned_peaks.append({
            "sample_index": orig_idx,
            "position": pos,
            "intensity": intensity,
            "note": "Not explained by any declared phase",
        })

    # Aggregate metrics
    if sample_peaks:
        result.overall_match_rate = round(total_matched / len(sample_peaks), 3)
    result.overall_grade = _grade_overall(result.overall_match_rate)

    return result
