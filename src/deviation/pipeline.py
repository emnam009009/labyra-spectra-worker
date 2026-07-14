"""
Deviation analysis pipeline orchestration.

Routing:
  - Sample has composition (multi-component): run match_multi_phase
  - Sample has only chemicalFormula (single phase): run single-phase match_peaks
  - Neither: skip deviation analysis

Returns dict ready to merge into analysisResult.

@phase R185-4-multi-phase-matcher (extends R185-3a)
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from src.deviation.composite_rules import run_composite_rules
from src.deviation.crystallinity import adaptive_tolerance, classify_crystallinity
from src.deviation.fraction_estimator import estimate_fractions
from src.deviation.multi_phase import (
    ComponentDeclaration,
    match_multi_phase,
)
from src.deviation.peak_matcher import (
    DEFAULT_TOLERANCES,
    SpectrumType,
    match_peaks,
)
from src.deviation.rules import run_rules

logger = logging.getLogger(__name__)

SIG_KEY_MAP: dict[str, str] = {
    "raman": "raman",
    "ftir": "ftir",
    "xrd": "xrd",
    "pl": "pl",
    "uvvis": "uvvis",
    "uvvis_drs": "uvvis",
}


def run_deviation_analysis(
    spectrum_type: str,
    parsed: dict[str, Any],
    material_profile: dict[str, Any] | None,
    laser_wavelength: int | None = None,
    composition: list[dict[str, Any]] | None = None,
    profile_loader: Callable[[str], dict[str, Any] | None] | None = None,
) -> dict[str, Any] | None:
    """
    Full deviation pipeline.

    Args:
        spectrum_type: raman/xrd/ftir/pl/uvvis(_drs)
        parsed: parser output with "peaks" list
        material_profile: single-phase reference (legacy path)
        laser_wavelength: 532/785/1064 for Raman context
        composition: NEW — list of {formula, role, nominal_fraction} from Sample
        profile_loader: NEW — callable(formula) -> profile dict, for multi-phase

    Returns:
        {
          "mode": "single-phase" | "multi-phase",
          "matchResult": ...,         # single-phase only
          "hypotheses": ...,          # single-phase only
          "multiPhase": ...,          # multi-phase only
          "referenceFormula": ...,
        }
        or None if cannot analyze.
    """
    if spectrum_type not in SIG_KEY_MAP:
        return None

    sample_peaks = parsed.get("peaks", [])
    if not sample_peaks:
        return None

    # ── Multi-phase path ─────────────────────────────────────────────────────
    if composition and len(composition) > 0 and profile_loader is not None:
        components = [
            ComponentDeclaration(
                formula=c["formula"],
                role=c.get("role", "matrix"),
                nominal_fraction=c.get("nominalFraction"),
                formation_method=c.get("formationMethod"),
            )
            for c in composition
            if c.get("formula")
        ]

        if components:
            matcher_type: SpectrumType = "uvvis" if spectrum_type == "uvvis_drs" else spectrum_type  # type: ignore[assignment]

            multi_result = match_multi_phase(
                sample_peaks=sample_peaks,
                components=components,
                profile_loader=profile_loader,
                spectrum_type=matcher_type,
            )

            # Run physics rules per-component (re-use single-phase rules)
            # Aggregated under "perComponentHypotheses"
            per_component_hyps: dict[str, list[dict[str, Any]]] = {}
            for cm in multi_result.components:
                # Reconstruct MatchResult from cm.match_result dict for run_rules
                from src.deviation.peak_matcher import (
                    MatchResult,
                    PeakMatch,
                    UnmatchedPeak,
                )
                mr_dict = cm.match_result
                # Rebuild MatchResult (dataclass) from dict
                matches = [PeakMatch(**m) for m in mr_dict["matches"]]
                unmatched_s = [UnmatchedPeak(**u) for u in mr_dict["unmatched_sample"]]
                unmatched_r = [UnmatchedPeak(**u) for u in mr_dict["unmatched_ref"]]
                reconstructed = MatchResult(
                    spectrum_type=mr_dict["spectrum_type"],
                    reference_formula=mr_dict["reference_formula"],
                    reference_label=mr_dict["reference_label"],
                    tolerance_used=mr_dict["tolerance_used"],
                    matches=matches,
                    unmatched_sample=unmatched_s,
                    unmatched_ref=unmatched_r,
                    match_count=mr_dict["match_count"],
                    match_rate=mr_dict["match_rate"],
                    mean_abs_deviation=mr_dict["mean_abs_deviation"],
                    max_abs_deviation=mr_dict["max_abs_deviation"],
                    rmse=mr_dict["rmse"],
                    quality_grade=mr_dict["quality_grade"],
                )
                hyps = run_rules(
                    reconstructed,
                    ctx={"laser_wavelength": laser_wavelength},
                )
                per_component_hyps[cm.formula] = [h.to_dict() for h in hyps]

            # R185-6: composite-specific rules (cross-phase phenomena)
            composite_hyps = run_composite_rules(
                multi_result,
                ctx={"laser_wavelength": laser_wavelength},
            )

            # R185-7: fraction estimation per component
            #   - XRD: RIR (if profiles have rirFactor) → quantitative
            #   - Raman: intensity ratio (qualitative only)
            #   - Others: peak-count fallback
            profile_dict = {
                c.formula: profile_loader(c.formula) or {}
                for c in multi_result.components
            }
            fraction_estimates = estimate_fractions(
                spectrum_type=spectrum_type,
                components=[c.to_dict() for c in multi_result.components],
                profiles=profile_dict,
            )

            return {
                "mode": "multi-phase",
                "multiPhase": multi_result.to_dict(),
                "perComponentHypotheses": per_component_hyps,
                "compositeHypotheses": [h.to_dict() for h in composite_hyps],
                "fractionEstimates": [fe.to_dict() for fe in fraction_estimates],
                "referenceFormula": None,
            }

    # ── Single-phase legacy path ─────────────────────────────────────────────
    if not material_profile:
        return None

    sig_key = SIG_KEY_MAP[spectrum_type]
    signatures = material_profile.get("spectralSignatures", {})
    sig = signatures.get(sig_key) if signatures else None
    if not sig or not sig.get("peaks"):
        return None

    formula = material_profile.get("formula") or material_profile.get("id", "")
    label = formula
    if material_profile.get("commonNames"):
        label = f"{formula} ({material_profile['commonNames'][0]})"

    matcher_type_sp: SpectrumType = "uvvis" if spectrum_type == "uvvis_drs" else spectrum_type  # type: ignore[assignment]

    # R185-5: First pass with default tolerance, then classify crystallinity,
    # then re-match if classification suggests broader tolerance is needed.
    base_tol = DEFAULT_TOLERANCES[matcher_type_sp]
    first_pass = match_peaks(
        sample_peaks=sample_peaks,
        ref_peaks=sig["peaks"],
        spectrum_type=matcher_type_sp,
        reference_formula=formula,
        reference_label=label,
        tolerance=base_tol,
    )

    crystal = classify_crystallinity(
        spectrum_type=spectrum_type,
        parsed=parsed,
        sample_peaks=sample_peaks,
        ref_peaks=sig["peaks"],
        matches=[m.__dict__ for m in first_pass.matches],
    )

    # Re-match with adaptive tolerance if classifier suggests nano/amorphous
    if crystal.tolerance_factor > 1.0:
        adjusted_tol = adaptive_tolerance(base_tol, crystal.classification)
        match_result = match_peaks(
            sample_peaks=sample_peaks,
            ref_peaks=sig["peaks"],
            spectrum_type=matcher_type_sp,
            reference_formula=formula,
            reference_label=label,
            tolerance=adjusted_tol,
        )
    else:
        match_result = first_pass

    laser = laser_wavelength or sig.get("laserWavelength")
    hypotheses = run_rules(
        match_result,
        ctx={"laser_wavelength": laser},
    )

    return {
        "mode": "single-phase",
        "matchResult": match_result.to_dict(),
        "crystallinity": crystal.to_dict(),
        "hypotheses": [h.to_dict() for h in hypotheses],
        "referenceFormula": formula,
        "referenceLabel": label,
        "referenceSource": material_profile.get("source"),
        "referenceMpId": material_profile.get("mpId"),
    }
