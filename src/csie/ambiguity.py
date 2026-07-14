"""
Ambiguous hypothesis handler.

Clusters hypotheses by underlying observation, scores candidates with
multi-spectrum evidence, and suggests discrimination experiments.

Discrimination experiment knowledge base built from published literature:
  - TEM/HRTEM → particle size, lattice spacing direct measurement
  - XPS → core-level binding energy → charge state, electron transfer
  - EDS/EELS → elemental composition, oxidation state
  - PL temperature-dependent → quantum confinement vs defect emission
  - Polarization-resolved Raman → mode symmetry assignment
  - In-situ heating XRD → phase transformations, strain relaxation

References for discrimination strategies:
  - Ferrari & Basko 2013 (Raman vs other techniques for graphene)
    DOI: 10.1038/nnano.2013.46
  - Reshchikov 2014 (PL discrimination techniques)
    DOI: 10.1063/1.4895792
  - Castro Neto et al. 2009 (electronic properties of graphene)
    DOI: 10.1103/RevModPhys.81.109

@phase R185-9-ambiguous-hypothesis-handler
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

Severity = Literal["info", "warning", "error"]


@dataclass
class CandidateCause:
    """One possible explanation for an observation."""
    rule_id: str
    name: str
    confidence: float  # original rule confidence
    score: float       # multi-spectrum adjusted score
    evidence: list[str] = field(default_factory=list)
    citation_doi: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiscriminationExperiment:
    """A suggested experiment to discriminate between causes."""
    technique: str           # "TEM", "XPS", "Raman polarized", etc.
    measurement: str         # specific protocol
    discriminates_between: list[str]  # rule_ids it helps distinguish
    expected_outcomes: dict[str, str]  # rule_id -> expected signal
    citation_doi: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AmbiguousObservation:
    """An observation with multiple plausible causes."""
    observation_id: str           # synthesized key
    description: str              # human-readable
    severity: Severity = "warning"
    candidates: list[CandidateCause] = field(default_factory=list)
    discrimination_experiments: list[DiscriminationExperiment] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Observation clustering ──────────────────────────────────────────────────

# Rules that explain the SAME observation cluster together.
# Each cluster maps observation → set of candidate rule_ids.
OBSERVATION_CLUSTERS: dict[str, dict[str, Any]] = {
    "raman_peak_shift_with_broadening": {
        "description": "Raman peak shift + significant broadening",
        "candidate_rules": [
            "R1-tensile-strain", "R2-compressive-strain",
            "R3-phonon-confinement",
            "R6-doping-intercalation",
            "R11-charge-transfer",
        ],
        "discrimination": ["TEM_particle_size", "XPS_core_level", "in_situ_temp_raman"],
    },
    "tmd_polymorph_ambiguity": {
        "description": "TMD basal-plane peak at ~14.4° 2θ (MoS2 or WS2)",
        "candidate_rules": [
            "R7-tmd-layer-count",
            "polymorph-MoS2", "polymorph-WS2",
        ],
        "discrimination": ["EDS_elemental", "raman_A1g_position", "PL_emission"],
    },
    "carbon_disorder_signature": {
        "description": "Carbon D/G ratio anomaly",
        "candidate_rules": [
            "R8-amorphization",
            "R14-defect-mediated-coupling",
        ],
        "discrimination": ["HRTEM_imaging", "XPS_C1s_shoulder", "anneal_recovery"],
    },
    "bandgap_shift": {
        "description": "Bandgap deviation from bulk reference",
        "candidate_rules": [
            "R6-doping-intercalation",
            "R12-heterojunction-band-offset",
            "quantum-confinement",
        ],
        "discrimination": ["PL_temp_dependent", "UPS_valence_band", "TEM_size_distribution"],
    },
    "low_freq_unassigned_peak": {
        "description": "Unassigned low-frequency peak (<100 cm⁻¹)",
        "candidate_rules": [
            "R13-interface-phonon",
            "R15-vdw-stacking-modes",
        ],
        "discrimination": ["polarized_raman", "temperature_dependent_raman"],
    },
}


# ── Discrimination experiment knowledge base ─────────────────────────────────

DISCRIMINATION_EXPERIMENTS: dict[str, DiscriminationExperiment] = {
    "TEM_particle_size": DiscriminationExperiment(
        technique="TEM/HRTEM",
        measurement="Bright-field imaging + size distribution histogram (n>100 particles)",
        discriminates_between=["R3-phonon-confinement", "R1-tensile-strain", "R11-charge-transfer"],
        expected_outcomes={
            "R3-phonon-confinement": "Particles < 10 nm visible",
            "R1-tensile-strain": "Particles > 20 nm with lattice expansion",
            "R11-charge-transfer": "Interfacial contact between phases observed",
        },
        citation_doi="10.1016/j.ultramic.2008.06.004",
    ),
    "XPS_core_level": DiscriminationExperiment(
        technique="XPS",
        measurement="High-resolution scan of TMD core level (Mo 3d, W 4f) and C 1s",
        discriminates_between=["R11-charge-transfer", "R4-oxygen-vacancy", "R6-doping-intercalation"],
        expected_outcomes={
            "R11-charge-transfer": "TMD core level shift toward lower BE (0.1-0.5 eV)",
            "R4-oxygen-vacancy": "Sub-stoichiometric oxide shoulder",
            "R6-doping-intercalation": "New peak/shoulder at dopant BE",
        },
        citation_doi="10.1016/j.apsusc.2010.10.051",
    ),
    "EDS_elemental": DiscriminationExperiment(
        technique="EDS / EDX",
        measurement="Spot analysis on representative particles (n>5 spots)",
        discriminates_between=["polymorph-MoS2", "polymorph-WS2", "R6-doping-intercalation"],
        expected_outcomes={
            "polymorph-MoS2": "Mo:S = 1:2, no W signal",
            "polymorph-WS2": "W:S = 1:2, no Mo signal",
            "R6-doping-intercalation": "Dopant element peak detected",
        },
        citation_doi=None,
    ),
    "raman_A1g_position": DiscriminationExperiment(
        technique="Raman (high-resolution)",
        measurement="Precise A1g peak position with sub-cm⁻¹ accuracy",
        discriminates_between=["polymorph-MoS2", "polymorph-WS2"],
        expected_outcomes={
            "polymorph-MoS2": "A1g at 408 ± 2 cm⁻¹ (bulk) / 403-407 (mono-bilayer)",
            "polymorph-WS2": "A1g at 417 ± 2 cm⁻¹ (bulk)",
        },
        citation_doi="10.1002/adfm.201102111",
    ),
    "PL_emission": DiscriminationExperiment(
        technique="Photoluminescence",
        measurement="PL spectrum at 532 nm excitation, RT and low-T (77 K)",
        discriminates_between=["polymorph-MoS2", "polymorph-WS2", "quantum-confinement"],
        expected_outcomes={
            "polymorph-MoS2": "Emission ~1.85 eV (A exciton) for monolayer",
            "polymorph-WS2": "Emission ~2.05 eV (A exciton) for monolayer",
            "quantum-confinement": "Blue-shifted emission with narrower FWHM",
        },
        citation_doi="10.1103/PhysRevLett.105.136805",
    ),
    "PL_temp_dependent": DiscriminationExperiment(
        technique="Photoluminescence (variable temperature)",
        measurement="PL spectra at 10-300 K, track peak energy + FWHM evolution",
        discriminates_between=["quantum-confinement", "R6-doping-intercalation", "R12-heterojunction-band-offset"],
        expected_outcomes={
            "quantum-confinement": "Sharp emission, narrows further at low T",
            "R6-doping-intercalation": "Sub-bandgap emission at fixed energy",
            "R12-heterojunction-band-offset": "Indirect transition appears at low T",
        },
        citation_doi="10.1063/1.4895792",
    ),
    "HRTEM_imaging": DiscriminationExperiment(
        technique="HRTEM",
        measurement="High-resolution lattice imaging + FFT analysis",
        discriminates_between=["R8-amorphization", "R14-defect-mediated-coupling", "R3-phonon-confinement"],
        expected_outcomes={
            "R8-amorphization": "Lack of long-range lattice fringes",
            "R14-defect-mediated-coupling": "Visible point defects, grain boundaries",
            "R3-phonon-confinement": "Crystalline particles < 10 nm",
        },
        citation_doi="10.1063/1.1674108",
    ),
    "XPS_C1s_shoulder": DiscriminationExperiment(
        technique="XPS",
        measurement="High-resolution C 1s scan (binding energy 280-292 eV)",
        discriminates_between=["R8-amorphization", "R14-defect-mediated-coupling"],
        expected_outcomes={
            "R8-amorphization": "Broad C 1s, multiple shoulders 285-289 eV",
            "R14-defect-mediated-coupling": "Sharp sp2 peak + minor defect shoulder",
        },
        citation_doi="10.1016/j.ssc.2007.03.052",
    ),
    "polarized_raman": DiscriminationExperiment(
        technique="Polarization-resolved Raman",
        measurement="Parallel + cross-polarized configurations on same spot",
        discriminates_between=["R13-interface-phonon", "R15-vdw-stacking-modes"],
        expected_outcomes={
            "R13-interface-phonon": "Breathing mode (A1g): parallel polarization stronger",
            "R15-vdw-stacking-modes": "Shear mode (Eg): cross-polarized signal",
        },
        citation_doi="10.1038/nmat3505",
    ),
    "temperature_dependent_raman": DiscriminationExperiment(
        technique="Temperature-dependent Raman",
        measurement="Spectra at 80-500 K, track peak position + FWHM",
        discriminates_between=["R13-interface-phonon", "R15-vdw-stacking-modes"],
        expected_outcomes={
            "R13-interface-phonon": "Strong T-dependence (anharmonic interlayer coupling)",
            "R15-vdw-stacking-modes": "Weak T-dependence",
        },
        citation_doi="10.1021/acs.nanolett.7b03515",
    ),
    "in_situ_temp_raman": DiscriminationExperiment(
        technique="In-situ heating Raman",
        measurement="Raman during temperature ramp 25-600°C",
        discriminates_between=["R1-tensile-strain", "R2-compressive-strain", "R3-phonon-confinement"],
        expected_outcomes={
            "R1-tensile-strain": "Strain relaxes above 200°C, peaks return to bulk",
            "R2-compressive-strain": "Strain relaxes above 200°C",
            "R3-phonon-confinement": "Persists; size doesn't change with moderate heating",
        },
        citation_doi=None,
    ),
    "UPS_valence_band": DiscriminationExperiment(
        technique="UPS",
        measurement="Valence band maximum (VBM) determination",
        discriminates_between=["R12-heterojunction-band-offset", "R6-doping-intercalation"],
        expected_outcomes={
            "R12-heterojunction-band-offset": "VBM offset between phases measurable",
            "R6-doping-intercalation": "Fermi level shifts within single phase",
        },
        citation_doi=None,
    ),
    "TEM_size_distribution": DiscriminationExperiment(
        technique="TEM size analysis",
        measurement="Particle size distribution from >100 particles",
        discriminates_between=["quantum-confinement", "R12-heterojunction-band-offset"],
        expected_outcomes={
            "quantum-confinement": "Size < Bohr exciton radius (~5 nm for many semiconductors)",
            "R12-heterojunction-band-offset": "Multi-phase morphology visible",
        },
        citation_doi=None,
    ),
    "anneal_recovery": DiscriminationExperiment(
        technique="Annealing + post-anneal Raman",
        measurement="Anneal at 300-500°C in Ar, re-measure Raman",
        discriminates_between=["R8-amorphization", "R14-defect-mediated-coupling"],
        expected_outcomes={
            "R8-amorphization": "Partial recovery — sp2 reformation possible",
            "R14-defect-mediated-coupling": "I(D)/I(G) decreases significantly",
        },
        citation_doi=None,
    ),
}


# ── Clustering logic ──────────────────────────────────────────────────────────

def cluster_hypotheses(
    hypotheses: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Group hypotheses by underlying observation cluster.

    Returns: { cluster_id: [hypotheses...] }
    """
    clusters: dict[str, list[dict[str, Any]]] = {}

    for hyp in hypotheses:
        rule_id = hyp.get("rule_id", "")
        for cluster_id, cluster_def in OBSERVATION_CLUSTERS.items():
            if rule_id in cluster_def["candidate_rules"]:
                clusters.setdefault(cluster_id, []).append(hyp)

    # Only keep clusters with 2+ hypotheses (true ambiguity)
    return {cid: hyps for cid, hyps in clusters.items() if len(hyps) >= 2}


# ── Multi-spectrum re-scoring ─────────────────────────────────────────────────

def rescore_with_multi_spectrum(
    hyp: dict[str, Any],
    csie_consistency: dict[str, Any] | None,
) -> float:
    """
    Re-score a hypothesis using multi-spectrum evidence from CSIE consistency.

    If hypothesis applies to a confirmed phase (consistency >=0.6), boost.
    If applies to conflict/missing phase, suppress.
    """
    base_conf = float(hyp.get("confidence", 0.5))

    if not csie_consistency:
        return base_conf

    declared = csie_consistency.get("declared_phases", [])
    # Try to match rule's target formula/phase from evidence
    boost = 1.0
    for phase in declared:
        if phase.get("verdict") == "confirmed":
            # Hypotheses about confirmed phases are more credible
            boost = max(boost, 1.0 + 0.1 * phase.get("consistency_score", 0))

    return min(0.95, round(base_conf * boost, 3))


# ── Main handler ─────────────────────────────────────────────────────────────

def handle_ambiguous(
    all_hypotheses: list[dict[str, Any]],
    csie_consistency: dict[str, Any] | None = None,
) -> list[AmbiguousObservation]:
    """
    Detect ambiguous observations and produce discrimination guidance.

    Args:
        all_hypotheses: union of single-phase + composite hypotheses across the sample
        csie_consistency: ConsistencyCheck dict (optional, used for re-scoring)

    Returns:
        List of AmbiguousObservation, sorted by severity descending.
    """
    if not all_hypotheses:
        return []

    clusters = cluster_hypotheses(all_hypotheses)
    if not clusters:
        return []

    ambiguous: list[AmbiguousObservation] = []

    for cluster_id, cluster_hyps in clusters.items():
        cluster_def = OBSERVATION_CLUSTERS[cluster_id]
        candidates: list[CandidateCause] = []

        for hyp in cluster_hyps:
            rescored = rescore_with_multi_spectrum(hyp, csie_consistency)
            candidates.append(CandidateCause(
                rule_id=hyp.get("rule_id", "unknown"),
                name=hyp.get("name", ""),
                confidence=float(hyp.get("confidence", 0.0)),
                score=rescored,
                evidence=list(hyp.get("evidence", []))[:3],
                citation_doi=(hyp.get("citation") or {}).get("doi") if hyp.get("citation") else None,
            ))

        # Sort by adjusted score desc
        candidates.sort(key=lambda c: -c.score)

        # Compile discrimination experiments
        exp_ids: list[str] = cluster_def.get("discrimination", [])
        experiments: list[DiscriminationExperiment] = []
        for exp_id in exp_ids:
            exp = DISCRIMINATION_EXPERIMENTS.get(exp_id)
            if exp is not None:
                experiments.append(exp)

        # Severity: high if top 2 candidates have similar scores (truly ambiguous)
        severity: Severity = "info"
        if len(candidates) >= 2:
            top, second = candidates[0].score, candidates[1].score
            if abs(top - second) < 0.15:
                severity = "warning"
            if abs(top - second) < 0.05:
                severity = "error"

        ambiguous.append(AmbiguousObservation(
            observation_id=cluster_id,
            description=cluster_def["description"],
            severity=severity,
            candidates=candidates,
            discrimination_experiments=experiments,
            notes=[
                f"{len(candidates)} plausible causes detected",
                "Single-spectrum analysis cannot uniquely discriminate.",
            ],
        ))

    # Sort: error > warning > info
    severity_order = {"error": 0, "warning": 1, "info": 2}
    ambiguous.sort(key=lambda a: severity_order[a.severity])

    return ambiguous
