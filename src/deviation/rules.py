"""
Physics rules for materials science deviation analysis.

Each rule is a pure function:
    fn(match_result: MatchResult, ctx: dict) -> Hypothesis | None

Rules are stateless and independently testable.

References — primary literature (DOI verified):
  R1/R2 strain — Williamson-Hall, Khorsand 2014 (Solid State Sciences)
  R3 phonon confinement — Bersani 1998 (PRB)
  R4 oxygen vacancy — Wang 2020 (Chem Mater) for WO3 specifically
  R7 TMD layer count — Lee 2010 (ACS Nano) for MoS2
  R8 amorphization — Tuinstra-Koenig 1970 (J Chem Phys) generalized

@phase R185-2-physics-rules-engine
"""
from __future__ import annotations

from typing import Any

from src.deviation.hypothesis import Hypothesis, RuleCitation
from src.deviation.peak_matcher import MatchResult, PeakMatch


# ── Citations ─────────────────────────────────────────────────────────────────

CIT_KHORSAND_STRAIN = RuleCitation(
    doi="10.1016/j.solidstatesciences.2014.04.012",
    journal="Solid State Sciences",
    year=2014,
    title="Williamson-Hall analysis in estimation of lattice strain",
)
CIT_BERSANI_PCM = RuleCitation(
    doi="10.1103/PhysRevB.63.125415",
    journal="Physical Review B",
    year=1998,
    title="Phonon confinement effects in TiO2 nanoparticles",
)
CIT_WANG_VO = RuleCitation(
    doi="10.1021/acs.chemmater.0c02029",
    journal="Chemistry of Materials",
    year=2020,
    title="Oxygen vacancy engineering in WO3",
)
CIT_LEE_MOS2 = RuleCitation(
    doi="10.1021/nl201874w",
    journal="Nano Letters",
    year=2012,
    title="Atomically thin MoS2: layer-dependent Raman shifts",
)
CIT_TUINSTRA_KOENIG = RuleCitation(
    doi="10.1063/1.1674108",
    journal="Journal of Chemical Physics",
    year=1970,
    title="Raman spectrum of graphite (D-band, disorder)",
)
CIT_FERRARI_GRAPHENE = RuleCitation(
    doi="10.1038/nnano.2013.46",
    journal="Nature Nanotechnology",
    year=2013,
    title="Raman spectroscopy as a versatile tool for studying graphene",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _signed_shifts(matches: list[PeakMatch]) -> list[float]:
    return [m.deviation for m in matches]


def _mean_fwhm_ratio(matches: list[PeakMatch], ref_fwhms: dict[int, float] | None = None) -> float:
    """Return mean(sample_fwhm / ref_fwhm) when ref_fwhm available, else 1.0."""
    if not ref_fwhms:
        return 1.0
    ratios = []
    for m in matches:
        ref_w = ref_fwhms.get(m.ref_index)
        if m.sample_fwhm and ref_w and ref_w > 0:
            ratios.append(m.sample_fwhm / ref_w)
    return sum(ratios) / len(ratios) if ratios else 1.0


def _has_consistent_shift_direction(shifts: list[float], min_count: int = 2) -> tuple[bool, str]:
    """Check if shifts are consistently positive or negative."""
    significant = [s for s in shifts if abs(s) >= 1.5]
    if len(significant) < min_count:
        return False, ""
    if all(s > 0 for s in significant):
        return True, "positive"
    if all(s < 0 for s in significant):
        return True, "negative"
    return False, ""


# ── R1: Tensile strain ────────────────────────────────────────────────────────

def rule_tensile_strain(result: MatchResult, ctx: dict[str, Any]) -> Hypothesis | None:
    """Tensile strain → mode softening (downshift). For Raman only."""
    if result.spectrum_type != "raman" or result.match_count < 2:
        return None

    shifts = _signed_shifts(result.matches)
    consistent, direction = _has_consistent_shift_direction(shifts)
    if not consistent or direction != "negative":
        return None

    mean_shift = sum(shifts) / len(shifts)
    if not (-8 < mean_shift < -1.5):
        return None

    # Rough estimate: dω/dε for typical TMD/oxide ~ -5 cm-1 / 1% strain
    strain_pct = abs(mean_shift / 5.0)
    fwhm_ratio = _mean_fwhm_ratio(result.matches, ctx.get("ref_fwhms"))

    evidence = [
        f"{result.match_count} peaks consistently downshifted (mean {mean_shift:.1f} cm-1)",
    ]
    if fwhm_ratio > 1.2:
        evidence.append(f"FWHM broadened ~{(fwhm_ratio - 1) * 100:.0f}% vs reference")

    confidence = 0.7 + (0.15 if fwhm_ratio > 1.2 else 0) + (0.10 if result.match_count >= 3 else 0)
    confidence = min(confidence, 0.95)

    return Hypothesis(
        rule_id="R1-tensile-strain",
        name="Tensile strain",
        confidence=round(confidence, 2),
        evidence=evidence,
        quantitative_estimate=f"Strain ~ {strain_pct:.1f}% (estimated)",
        suggested_followup="Confirm via XRD: expect (002) peak downshift by ~0.05-0.1 deg.",
        citation=CIT_KHORSAND_STRAIN,
        severity="notice",
    )


# ── R2: Compressive strain ────────────────────────────────────────────────────

def rule_compressive_strain(result: MatchResult, ctx: dict[str, Any]) -> Hypothesis | None:
    """Compressive strain → mode stiffening (upshift)."""
    if result.spectrum_type != "raman" or result.match_count < 2:
        return None

    shifts = _signed_shifts(result.matches)
    consistent, direction = _has_consistent_shift_direction(shifts)
    if not consistent or direction != "positive":
        return None

    mean_shift = sum(shifts) / len(shifts)
    if not (1.5 < mean_shift < 8):
        return None

    strain_pct = abs(mean_shift / 5.0)
    fwhm_ratio = _mean_fwhm_ratio(result.matches, ctx.get("ref_fwhms"))

    evidence = [
        f"{result.match_count} peaks consistently upshifted (mean +{mean_shift:.1f} cm-1)",
    ]
    if fwhm_ratio > 1.2:
        evidence.append(f"FWHM broadened ~{(fwhm_ratio - 1) * 100:.0f}% vs reference")

    confidence = min(0.7 + (0.15 if fwhm_ratio > 1.2 else 0) + (0.10 if result.match_count >= 3 else 0), 0.95)

    return Hypothesis(
        rule_id="R2-compressive-strain",
        name="Compressive strain",
        confidence=round(confidence, 2),
        evidence=evidence,
        quantitative_estimate=f"Strain ~ {strain_pct:.1f}% (estimated)",
        suggested_followup="Confirm via XRD: expect (002) peak upshift by ~0.05-0.1 deg.",
        citation=CIT_KHORSAND_STRAIN,
        severity="notice",
    )


# ── R3: Phonon confinement (nanoparticle) ─────────────────────────────────────

def rule_phonon_confinement(result: MatchResult, ctx: dict[str, Any]) -> Hypothesis | None:
    """
    Phonon confinement model (PCM): small particle (<10 nm) causes:
      - Asymmetric broadening (FWHM 1.5x+)
      - Small upshift (1-4 cm-1) for most oxides
    Distinguishes from compressive strain by FWHM ratio threshold.
    """
    if result.spectrum_type != "raman" or result.match_count < 1:
        return None

    fwhm_ratio = _mean_fwhm_ratio(result.matches, ctx.get("ref_fwhms"))
    if fwhm_ratio < 1.5:
        return None  # broadening must be significant

    shifts = _signed_shifts(result.matches)
    mean_shift = sum(shifts) / len(shifts) if shifts else 0

    # PCM shift is small (<5 cm-1), distinguishing from strain
    if abs(mean_shift) > 5:
        return None

    # Estimate particle size from FWHM ratio (very rough)
    # For TiO2 anatase: FWHM ratio ~2 → ~5nm, ratio ~3 → ~3nm
    estimated_size_nm = round(10 / (fwhm_ratio - 0.5), 1)

    return Hypothesis(
        rule_id="R3-phonon-confinement",
        name="Phonon confinement (nanoparticle)",
        confidence=round(0.6 + min(0.25, (fwhm_ratio - 1.5) * 0.2), 2),
        evidence=[
            f"Significant FWHM broadening: {fwhm_ratio:.1f}x vs reference",
            f"Small/moderate peak shift ({mean_shift:+.1f} cm-1) inconsistent with strain",
        ],
        quantitative_estimate=f"Particle size ~ {estimated_size_nm} nm (rough PCM estimate)",
        suggested_followup="Confirm via TEM/SEM particle size distribution.",
        citation=CIT_BERSANI_PCM,
        severity="info",
    )


# ── R4: Oxygen vacancy (oxide-specific) ───────────────────────────────────────

def rule_oxygen_vacancy(result: MatchResult, ctx: dict[str, Any]) -> Hypothesis | None:
    """
    Oxygen vacancy signature: typically observed via combination of
    Raman softening + lattice expansion (XRD) + UV-Vis redshift.

    This rule fires on Raman alone but flags multi-spectrum corroboration needed.
    Applicable to metal oxides (formula contains 'O' and has 'oxide' material class).
    """
    formula = result.reference_formula
    if "O" not in formula:
        return None
    # Skip non-oxides like SiO2 (not transition metal oxide)
    transition_oxides = {"WO3", "TiO2", "ZnO", "Fe2O3", "MoO3", "V2O5", "MnO2", "NiO", "CoO", "CuO"}
    if formula not in transition_oxides:
        return None

    if result.spectrum_type != "raman" or result.match_count < 2:
        return None

    shifts = _signed_shifts(result.matches)
    consistent, direction = _has_consistent_shift_direction(shifts)
    if not consistent or direction != "negative":
        return None

    mean_shift = sum(shifts) / len(shifts)
    if abs(mean_shift) < 2:
        return None

    return Hypothesis(
        rule_id="R4-oxygen-vacancy",
        name="Oxygen vacancy doping",
        confidence=0.65,
        evidence=[
            f"Metal oxide ({formula}) with consistent Raman softening (mean {mean_shift:.1f} cm-1)",
            "Mode softening consistent with M-O bond weakening from reduced cation",
        ],
        quantitative_estimate=None,
        suggested_followup=(
            "Multi-spectrum corroboration recommended: "
            "(1) XPS for cation oxidation state shoulder, "
            "(2) UV-Vis for sub-bandgap absorption, "
            "(3) EPR for unpaired electrons. "
            "Single-spectrum diagnosis insufficient."
        ),
        citation=CIT_WANG_VO,
        severity="warning",
    )


# ── R5: Mixed phase contamination ─────────────────────────────────────────────

def rule_mixed_phase(result: MatchResult, ctx: dict[str, Any]) -> Hypothesis | None:
    """High primary phase match + 2+ strong unmatched sample peaks → secondary phase."""
    if result.match_rate < 0.5:
        return None  # primary phase not well-matched; not "mixed" but "wrong ref"

    strong_unmatched = [u for u in result.unmatched_sample if u.intensity > 20]
    if len(strong_unmatched) < 2:
        return None

    # Format peak positions for evidence
    positions_str = ", ".join(f"{u.position:.0f}" for u in strong_unmatched[:5])

    return Hypothesis(
        rule_id="R5-mixed-phase",
        name="Mixed phase contamination",
        confidence=0.70,
        evidence=[
            f"Primary phase ({result.reference_label}) matches at {result.match_rate * 100:.0f}%",
            f"{len(strong_unmatched)} strong unmatched peak(s) at: {positions_str}",
        ],
        quantitative_estimate=None,
        suggested_followup=(
            "Search reference library for matching secondary phase. "
            "Common candidates: polymorphs of same formula (e.g., m-WO3 + h-WO3, "
            "anatase + rutile TiO2), or precursor residue."
        ),
        citation=None,
        severity="warning",
    )


# ── R6: Doping / intercalation ────────────────────────────────────────────────

def rule_doping_intercalation(result: MatchResult, ctx: dict[str, Any]) -> Hypothesis | None:
    """
    Foreign atom intercalation or doping signature:
      - Asymmetric peak broadening (some peaks broaden, others not)
      - 1+ new low-frequency mode (<200 cm-1 for Raman)
    """
    if result.spectrum_type != "raman":
        return None

    low_freq_unmatched = [
        u for u in result.unmatched_sample
        if u.position < 200 and u.intensity > 15
    ]
    if not low_freq_unmatched:
        return None

    # Check FWHM variance — doping causes non-uniform broadening
    fwhms = [m.sample_fwhm for m in result.matches if m.sample_fwhm]
    if len(fwhms) < 2:
        return None

    fwhm_mean = sum(fwhms) / len(fwhms)
    fwhm_var = sum((f - fwhm_mean) ** 2 for f in fwhms) / len(fwhms)
    fwhm_cv = (fwhm_var ** 0.5) / fwhm_mean if fwhm_mean > 0 else 0

    if fwhm_cv < 0.3:
        return None  # uniform broadening rules out non-uniform doping

    return Hypothesis(
        rule_id="R6-doping-intercalation",
        name="Doping or intercalation",
        confidence=0.55,
        evidence=[
            f"{len(low_freq_unmatched)} new low-frequency mode(s) below 200 cm-1",
            f"Non-uniform peak broadening (CV = {fwhm_cv:.2f})",
        ],
        suggested_followup=(
            "Identify dopant via XPS or EDS. "
            "Low-freq modes often indicate heavy-atom intercalation (alkali, lanthanide)."
        ),
        citation=None,
        severity="info",
    )


# ── R7: TMD layer count (MoS2, WS2, MoSe2 specific) ───────────────────────────

def rule_tmd_layer_count(result: MatchResult, ctx: dict[str, Any]) -> Hypothesis | None:
    """For TMDs: Δ(A1g - E12g) indicates layer count."""
    if result.spectrum_type != "raman":
        return None
    if result.reference_formula not in ("MoS2", "WS2", "MoSe2", "WSe2"):
        return None
    if result.match_count < 2:
        return None

    # Find E12g (in-plane) and A1g (out-of-plane) by assignment text
    e12g = None
    a1g = None
    for m in result.matches:
        assignment_lower = m.ref_assignment.lower()
        if "e12g" in assignment_lower or "e^1" in assignment_lower or "in-plane" in assignment_lower:
            e12g = m
        elif "a1g" in assignment_lower or "out-of-plane" in assignment_lower:
            a1g = m

    if not e12g or not a1g:
        return None

    delta = a1g.sample_position - e12g.sample_position

    # MoS2 layer mapping (Lee 2010, ACS Nano)
    if result.reference_formula == "MoS2":
        if 18 <= delta <= 21:
            layer = "monolayer"
            confidence = 0.90
        elif 21 < delta <= 22:
            layer = "bilayer"
            confidence = 0.85
        elif 22 < delta <= 23:
            layer = "trilayer"
            confidence = 0.80
        elif 23 < delta <= 25:
            layer = "few-layer (4-6)"
            confidence = 0.75
        elif delta > 25:
            layer = "bulk (>6 layers)"
            confidence = 0.85
        else:
            layer = "unusual (sub-monolayer?)"
            confidence = 0.40
    else:
        # Generic for WS2/MoSe2/WSe2 — broader brackets
        if delta < 62:
            layer = "monolayer"
            confidence = 0.75
        elif delta > 65:
            layer = "bulk"
            confidence = 0.75
        else:
            layer = "few-layer"
            confidence = 0.65

    return Hypothesis(
        rule_id="R7-tmd-layer-count",
        name=f"TMD layer count: {layer}",
        confidence=confidence,
        evidence=[
            f"A1g at {a1g.sample_position:.1f} cm-1, E12g at {e12g.sample_position:.1f} cm-1",
            f"Peak separation Δ = {delta:.1f} cm-1",
        ],
        quantitative_estimate=f"Estimated layer count: {layer}",
        suggested_followup="Confirm via AFM thickness or HRTEM cross-section.",
        citation=CIT_LEE_MOS2,
        severity="info",
    )


# ── R8: Amorphization / crystallinity loss ───────────────────────────────────

def rule_amorphization(result: MatchResult, ctx: dict[str, Any]) -> Hypothesis | None:
    """All peaks uniformly broadened + low intensity → amorphous tendency."""
    if result.match_count < 2:
        return None

    fwhm_ratio = _mean_fwhm_ratio(result.matches, ctx.get("ref_fwhms"))
    if fwhm_ratio < 2.0:
        return None  # need significant broadening

    # Check uniform broadening (low CV)
    fwhms = [m.sample_fwhm for m in result.matches if m.sample_fwhm]
    if len(fwhms) < 2:
        return None
    fwhm_mean = sum(fwhms) / len(fwhms)
    fwhm_var = sum((f - fwhm_mean) ** 2 for f in fwhms) / len(fwhms)
    fwhm_cv = (fwhm_var ** 0.5) / fwhm_mean if fwhm_mean > 0 else 0

    if fwhm_cv > 0.25:
        return None  # non-uniform broadening = doping, not amorphization

    return Hypothesis(
        rule_id="R8-amorphization",
        name="Crystallinity loss (amorphization tendency)",
        confidence=0.70,
        evidence=[
            f"All peaks broadened ~{fwhm_ratio:.1f}x uniformly (CV = {fwhm_cv:.2f})",
        ],
        quantitative_estimate=None,
        suggested_followup=(
            "Confirm via XRD: expect peak broadening + amorphous halo. "
            "Consider annealing to restore crystallinity."
        ),
        citation=CIT_TUINSTRA_KOENIG,
        severity="warning",
    )


# ── R9: Substrate signature ──────────────────────────────────────────────────

# Common substrate Raman peaks (cm-1)
SUBSTRATE_SIGNATURES: dict[str, tuple[str, float]] = {
    # peak position : (substrate name, tolerance)
}
SUBSTRATE_PEAKS = [
    (520, "Si (c-Si)", 3),
    (437, "ZnO (wurtzite E2 high)", 3),
    (1095, "Cellulose (paper substrate)", 5),
    (1728, "PMMA (acrylic mount)", 5),
    (1580, "Graphite/HOPG substrate G-band", 5),
]


def rule_substrate_contribution(result: MatchResult, ctx: dict[str, Any]) -> Hypothesis | None:
    """Unmatched peaks at known substrate Raman positions."""
    if result.spectrum_type != "raman":
        return None

    suspected = []
    for u in result.unmatched_sample:
        for sub_pos, sub_name, tol in SUBSTRATE_PEAKS:
            if abs(u.position - sub_pos) <= tol and u.intensity > 10:
                suspected.append((sub_name, u.position, sub_pos))
                break

    if not suspected:
        return None

    evidence_lines = [
        f"Unmatched peak at {pos:.0f} cm-1 matches {name} (expected {ref:.0f})"
        for name, pos, ref in suspected
    ]

    return Hypothesis(
        rule_id="R9-substrate-contribution",
        name="Substrate contribution",
        confidence=0.80,
        evidence=evidence_lines,
        quantitative_estimate=None,
        suggested_followup=(
            "These peaks may be from substrate, not sample. "
            "Re-measure with substrate background subtraction, or use confocal "
            "Raman with smaller z-depth to isolate sample layer."
        ),
        citation=None,
        severity="warning",
    )


# ── R10: Resonance enhancement (WS2 532nm specific) ──────────────────────────

def rule_resonance_enhancement(result: MatchResult, ctx: dict[str, Any]) -> Hypothesis | None:
    """
    WS2 at 532 nm: 2LA(M) mode ~350 cm-1 strongly resonance-enhanced.
    If sample shows strong unmatched peak at 350 cm-1, this is NOT a defect.
    """
    if result.spectrum_type != "raman" or result.reference_formula != "WS2":
        return None

    laser = ctx.get("laser_wavelength")
    if laser != 532:
        return None

    for u in result.unmatched_sample:
        if 346 <= u.position <= 354 and u.intensity > 25:
            return Hypothesis(
                rule_id="R10-resonance-enhancement",
                name="Resonance Raman: 2LA(M) mode",
                confidence=0.85,
                evidence=[
                    f"Strong unmatched peak at {u.position:.0f} cm-1 with 532 nm excitation",
                    "WS2 has known resonance enhancement of 2LA(M) ~350 cm-1 at 532 nm",
                ],
                quantitative_estimate=None,
                suggested_followup=(
                    "This is a resonance feature, NOT a defect. "
                    "Use 785 nm laser to suppress and obtain non-resonant spectrum."
                ),
                citation=RuleCitation(
                    doi="10.1021/acs.nanolett.5b01925",
                    journal="Nano Letters",
                    year=2015,
                    title="Probing interlayer coupling in WS2 via Raman spectroscopy",
                ),
                severity="info",
            )
    return None


# ── Engine ────────────────────────────────────────────────────────────────────

ALL_RULES = [
    rule_tensile_strain,
    rule_compressive_strain,
    rule_phonon_confinement,
    rule_oxygen_vacancy,
    rule_mixed_phase,
    rule_doping_intercalation,
    rule_tmd_layer_count,
    rule_amorphization,
    rule_substrate_contribution,
    rule_resonance_enhancement,
]


def run_rules(
    result: MatchResult,
    ctx: dict[str, Any] | None = None,
) -> list[Hypothesis]:
    """
    Run all physics rules against a MatchResult, return hypotheses sorted
    by confidence descending.

    ctx may contain:
      - laser_wavelength: int (532/785/1064) for Raman context
      - ref_fwhms: dict[ref_index, float] for FWHM-based rules
    """
    ctx = ctx or {}
    hypotheses: list[Hypothesis] = []
    for rule_fn in ALL_RULES:
        try:
            h = rule_fn(result, ctx)
            if h is not None:
                hypotheses.append(h)
        except Exception:  # noqa: BLE001
            # Rule failures should not break the engine
            import logging
            logging.getLogger(__name__).exception("Rule %s raised", rule_fn.__name__)

    # Sort by confidence descending
    hypotheses.sort(key=lambda h: -h.confidence)
    return hypotheses
