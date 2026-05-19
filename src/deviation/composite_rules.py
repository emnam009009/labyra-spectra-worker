"""
Composite-specific physics rules (R11-R15).

These rules detect phenomena that only exist in multi-phase samples:
charge transfer, heterojunction band offset, interface phonon modes,
defect-mediated coupling, vdW stacking modes.

Each rule takes a MultiPhaseResult and optional context, returns a
Hypothesis or None. Rules are pure functions, individually testable.

References — all peer-reviewed:
  R11 charge transfer: Chen et al. (2014). DOI: 10.1021/nn5025654
  R12 band offset: Xu et al. (2018). DOI: 10.1038/s41467-018-04748-x
  R13 interface phonon: Lin et al. (2017). DOI: 10.1021/acs.nanolett.7b03515
  R14 defect coupling: Ferrari 2013 [Nature Nanotech]. DOI: 10.1038/nnano.2013.46
  R15 vdW breathing: Tan et al. (2012). DOI: 10.1038/nmat3505

@phase R185-6-composite-rules
"""
from __future__ import annotations

from typing import Any

from src.deviation.hypothesis import Hypothesis, RuleCitation
from src.deviation.multi_phase import ComponentMatch, MultiPhaseResult


# ── Citations ─────────────────────────────────────────────────────────────────

CIT_CHEN_CHARGE_TRANSFER = RuleCitation(
    doi="10.1021/nn5025654",
    journal="ACS Nano",
    year=2014,
    title="Charge transfer in MoS2/graphene heterostructures via Raman",
)
CIT_XU_BAND_OFFSET = RuleCitation(
    doi="10.1038/s41467-018-04748-x",
    journal="Nature Communications",
    year=2018,
    title="Type-II band alignment in TMD heterostructures",
)
CIT_LIN_INTERFACE_PHONON = RuleCitation(
    doi="10.1021/acs.nanolett.7b03515",
    journal="Nano Letters",
    year=2017,
    title="Interlayer breathing modes in vdW heterostructures",
)
CIT_FERRARI_DEFECT = RuleCitation(
    doi="10.1038/nnano.2013.46",
    journal="Nature Nanotechnology",
    year=2013,
    title="Raman spectroscopy as versatile tool for graphene defects",
)
CIT_TAN_VDW = RuleCitation(
    doi="10.1038/nmat3505",
    journal="Nature Materials",
    year=2012,
    title="The shear mode of multilayer graphene",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_component(result: MultiPhaseResult, formula: str) -> ComponentMatch | None:
    """Return ComponentMatch for given formula, or None."""
    for c in result.components:
        if c.formula == formula:
            return c
    return None


def _component_peak(comp: ComponentMatch, assignment_substr: str) -> dict[str, Any] | None:
    """Find a matched peak by partial assignment label match (case-insensitive)."""
    matches = comp.match_result.get("matches", [])
    for m in matches:
        if assignment_substr.lower() in (m.get("ref_assignment", "") or "").lower():
            return m
    return None


def _has_composite_formulas(result: MultiPhaseResult, *required: str) -> bool:
    """Check if all required formulas are observed (not just declared)."""
    observed = {
        c.formula for c in result.components
        if c.intent_coverage > 0.3
    }
    return all(f in observed for f in required)


# ── R11: Charge transfer in TMD/graphene heterostructures ────────────────────

def rule_charge_transfer_tmd_graphene(
    result: MultiPhaseResult,
    ctx: dict[str, Any],
) -> Hypothesis | None:
    """
    TMD/graphene charge transfer signature (Chen 2014):
      - G-band UPSHIFT in graphene/rGO (electron withdrawal)
      - A1g DOWNSHIFT in TMD (electron accumulation)

    Requires both TMD and carbon components observed.
    """
    if result.spectrum_type != "raman":
        return None

    tmd_formulas = ["MoS2", "WS2", "MoSe2", "WSe2"]
    tmd_comp = next(
        (c for c in result.components if c.formula in tmd_formulas and c.intent_coverage > 0.3),
        None,
    )
    carbon_comp = _get_component(result, "C")
    if not tmd_comp or not carbon_comp or carbon_comp.intent_coverage <= 0.3:
        return None

    # Find A1g shift in TMD
    a1g_peak = _component_peak(tmd_comp, "a1g")
    g_peak = _component_peak(carbon_comp, "g-band")
    if not a1g_peak or not g_peak:
        return None

    a1g_dev = a1g_peak.get("deviation", 0)
    g_dev = g_peak.get("deviation", 0)

    # Signature: A1g down + G up
    if a1g_dev >= -0.5 or g_dev <= 0.5:
        return None

    evidence = [
        f"TMD A1g downshifted by {a1g_dev:.1f} cm-1 (electron accumulation)",
        f"Carbon G-band upshifted by +{g_dev:.1f} cm-1 (electron donation)",
        f"Charge transfer signature consistent across both phases",
    ]

    confidence = min(0.85, 0.6 + min(abs(a1g_dev), 5) * 0.05 + min(g_dev, 5) * 0.03)

    return Hypothesis(
        rule_id="R11-charge-transfer",
        name=f"Charge transfer ({tmd_comp.formula} -> C)",
        confidence=round(confidence, 2),
        evidence=evidence,
        quantitative_estimate=(
            f"Electron transfer ~ 10^12-10^13 cm-2 (order of magnitude). "
            f"Confirm via XPS binding energy shifts."
        ),
        suggested_followup=(
            "XPS: shift in TMD core level (e.g. Mo 3d, W 4f) toward lower BE. "
            "PL quenching also supports inter-layer charge transfer."
        ),
        citation=CIT_CHEN_CHARGE_TRANSFER,
        severity="info",
    )


# ── R12: Heterojunction band offset via UV-Vis ──────────────────────────────

def rule_heterojunction_band_offset(
    result: MultiPhaseResult,
    ctx: dict[str, Any],
) -> Hypothesis | None:
    """
    UV-Vis composite edge shift relative to constituent components.

    Type-II band alignment: composite absorption edge red-shifts
    vs each individual component due to staggered band offset.
    """
    if result.spectrum_type != "uvvis":
        return None

    # Need 2+ semiconductor components observed
    observed = [c for c in result.components if c.intent_coverage > 0.3]
    if len(observed) < 2:
        return None

    # Look for absorption edge match deviation
    # Simplified: if any component shows significant negative deviation
    # in UV-Vis peaks, flag as possible heterojunction signature.
    significant_redshifts = 0
    components_with_redshift = []
    for c in observed:
        for m in c.match_result.get("matches", []):
            dev = m.get("deviation", 0)
            if dev < -10:  # nm scale redshift
                significant_redshifts += 1
                components_with_redshift.append(c.formula)
                break

    if significant_redshifts < 2:
        return None

    return Hypothesis(
        rule_id="R12-heterojunction-band-offset",
        name="Heterojunction band offset",
        confidence=0.60,
        evidence=[
            f"Absorption edges red-shifted in: {", ".join(set(components_with_redshift))}",
            "Pattern consistent with Type-II band alignment",
        ],
        quantitative_estimate=None,
        suggested_followup=(
            "Confirm via PL: heterostructure should show suppressed direct emission "
            "and possibly indirect inter-layer emission at lower energy. "
            "UPS/XPS for valence band offset measurement."
        ),
        citation=CIT_XU_BAND_OFFSET,
        severity="info",
    )


# ── R13: Interface phonon mode (vdW gap breathing) ──────────────────────────

def rule_interface_phonon_mode(
    result: MultiPhaseResult,
    ctx: dict[str, Any],
) -> Hypothesis | None:
    """
    vdW heterostructure interface phonon: NEW low-frequency Raman mode
    that does not belong to either constituent (typically <100 cm-1).

    Signature: at least 2 phases observed AND unassigned peak < 100 cm-1
    with reasonable intensity.
    """
    if result.spectrum_type != "raman":
        return None
    if len(result.components) < 2:
        return None

    # Need 2+ phases observed
    n_observed = sum(1 for c in result.components if c.intent_coverage > 0.3)
    if n_observed < 2:
        return None

    # Look for unassigned peaks in low-freq region
    interface_candidates = [
        u for u in result.unassigned_peaks
        if 20 < u["position"] < 100 and u["intensity"] > 10
    ]

    if not interface_candidates:
        return None

    positions_str = ", ".join(f"{u["position"]:.0f}" for u in interface_candidates[:3])

    return Hypothesis(
        rule_id="R13-interface-phonon",
        name="Interface phonon (vdW breathing mode)",
        confidence=0.70,
        evidence=[
            f"{n_observed} phases observed simultaneously",
            f"Unassigned low-freq peak(s) at: {positions_str} cm-1",
            "Consistent with vdW gap breathing or shear mode in heterostructure",
        ],
        quantitative_estimate=(
            "Layer breathing modes are diagnostic of layer count. "
            "Frequency scales as 1/sqrt(N) for N-layer stack."
        ),
        suggested_followup=(
            "Polarization-resolved Raman to distinguish breathing (Ag) vs shear (Eg) modes. "
            "Low-T measurement to sharpen peaks."
        ),
        citation=CIT_LIN_INTERFACE_PHONON,
        severity="info",
    )


# ── R14: Defect-mediated coupling (D/G ratio modulation) ────────────────────

def rule_defect_mediated_coupling(
    result: MultiPhaseResult,
    ctx: dict[str, Any],
) -> Hypothesis | None:
    """
    For carbon component in composite: D/G intensity ratio modulation
    indicates defect engineering or composite-induced disorder.

    Compares D-band and G-band intensities in carbon component.
    Significant deviation from typical rGO baseline (I_D/I_G ~ 1) suggests
    composite-induced restructuring of carbon network.
    """
    if result.spectrum_type != "raman":
        return None

    carbon_comp = _get_component(result, "C")
    if not carbon_comp or carbon_comp.intent_coverage <= 0.5:
        return None

    d_peak = _component_peak(carbon_comp, "d-band")
    g_peak = _component_peak(carbon_comp, "g-band")
    if not d_peak or not g_peak:
        return None

    d_int = d_peak.get("sample_intensity", 0)
    g_int = g_peak.get("sample_intensity", 0)
    if g_int <= 0:
        return None

    dg_ratio = d_int / g_int

    # Significantly different from typical rGO (0.8-1.2) or pristine graphene (<0.1)
    if 0.5 < dg_ratio < 1.5:
        # Normal range, no alert
        return None

    if dg_ratio > 1.5:
        evidence_text = f"I(D)/I(G) = {dg_ratio:.2f} (highly defective carbon)"
        followup_text = (
            "High disorder. Confirm via XPS C 1s shoulder, "
            "consider annealing to recover sp2 network."
        )
        confidence = min(0.80, 0.55 + (dg_ratio - 1.5) * 0.1)
    else:  # < 0.5
        evidence_text = f"I(D)/I(G) = {dg_ratio:.2f} (low defects — near-pristine)"
        followup_text = "Verify via HRTEM whether graphitic ordering is intact."
        confidence = 0.65

    return Hypothesis(
        rule_id="R14-defect-mediated-coupling",
        name="Carbon defect-mediated coupling",
        confidence=round(confidence, 2),
        evidence=[
            evidence_text,
            f"Carbon component in composite with {len(result.components) - 1} other phase(s)",
        ],
        quantitative_estimate=f"I(D)/I(G) = {dg_ratio:.2f}",
        suggested_followup=followup_text,
        citation=CIT_FERRARI_DEFECT,
        severity="info",
    )


# ── R15: vdW stacking / shear modes ─────────────────────────────────────────

def rule_vdw_stacking_modes(
    result: MultiPhaseResult,
    ctx: dict[str, Any],
) -> Hypothesis | None:
    """
    Layer breathing / shear modes <50 cm-1 indicate well-defined vdW stacking.

    Different from R13 (interface phonon 20-100 cm-1): R15 specifically
    targets ultra-low-frequency modes <50 cm-1 in TMDs that confirm
    proper layered structure.
    """
    if result.spectrum_type != "raman":
        return None

    # Only meaningful for 2D materials
    tmd_formulas = {"MoS2", "WS2", "MoSe2", "WSe2", "C"}
    has_2d = any(
        c.formula in tmd_formulas and c.intent_coverage > 0.3
        for c in result.components
    )
    if not has_2d:
        return None

    ultra_low_freq = [
        u for u in result.unassigned_peaks
        if u["position"] < 50 and u["intensity"] > 5
    ]
    if not ultra_low_freq:
        return None

    return Hypothesis(
        rule_id="R15-vdw-stacking-modes",
        name="vdW stacking modes (low-freq Raman)",
        confidence=0.65,
        evidence=[
            f"Ultra-low frequency peak(s) at: {", ".join(f'{u["position"]:.0f}' for u in ultra_low_freq)} cm-1",
            "2D material component(s) detected in sample",
        ],
        quantitative_estimate=(
            "Frequency relates to layer count via 1/sqrt(N) scaling. "
            "Requires polarization analysis for definitive assignment."
        ),
        suggested_followup=(
            "Polarization-resolved Raman: shear modes are Eg (cross-polarized), "
            "breathing modes are A1g (parallel)."
        ),
        citation=CIT_TAN_VDW,
        severity="info",
    )


# ── Engine ───────────────────────────────────────────────────────────────────

COMPOSITE_RULES = [
    rule_charge_transfer_tmd_graphene,
    rule_heterojunction_band_offset,
    rule_interface_phonon_mode,
    rule_defect_mediated_coupling,
    rule_vdw_stacking_modes,
]


def run_composite_rules(
    result: MultiPhaseResult,
    ctx: dict[str, Any] | None = None,
) -> list[Hypothesis]:
    """Run composite rules against MultiPhaseResult. Sorted by confidence."""
    ctx = ctx or {}
    hypotheses: list[Hypothesis] = []
    for rule_fn in COMPOSITE_RULES:
        try:
            h = rule_fn(result, ctx)
            if h is not None:
                hypotheses.append(h)
        except Exception:  # noqa: BLE001
            import logging
            logging.getLogger(__name__).exception("Composite rule %s raised", rule_fn.__name__)
    hypotheses.sort(key=lambda h: -h.confidence)
    return hypotheses
