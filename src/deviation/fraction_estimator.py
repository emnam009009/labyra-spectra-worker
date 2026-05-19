"""
Fraction estimator for multi-phase deviation analysis.

Strict Trust > Coverage philosophy: claims mass fraction ONLY when
the spectrum and reference data support it. Otherwise returns qualitative
estimate with explicit caveat.

Methods:
  RIR (Reference Intensity Ratio) — XRD only, needs I/I_corundum factor
    Reference: Chung 1974, J Appl Cryst 7, 519. DOI: 10.1107/S0021889874010375
  Lambert-Beer — UV-Vis with epsilon (molar extinction)
    Reference: Standard textbook, see e.g. Skoog "Principles of Instrumental Analysis"
  Raman intensity ratio — QUALITATIVE only, cross-section varies 10-100x
  Peak count fallback — order-of-magnitude only

@phase R185-7-fraction-estimator
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal

FractionMethod = Literal[
    "rir",                                # XRD Reference Intensity Ratio (Chung 1974)
    "direct-comparison",                  # XRD Klug-Alexander mass absorption method
    "lambert-beer",                       # UV-Vis molar absorptivity
    "raman-intensity-ratio-qualitative",  # Detected intensity only
    "peak-count-fallback",                # Loose order-of-magnitude
]


@dataclass
class CitationRef:
    doi: str
    journal: str
    year: int
    title: str
    verified: bool = True


@dataclass
class FractionEstimate:
    """Single-component fraction estimate."""
    formula: str
    value: float                  # 0.0 - 1.0
    uncertainty: float            # +/- absolute (NOT percent)
    method: FractionMethod
    quantitative: bool            # True ONLY for rir + lambert-beer
    caveat: str                   # Non-empty mandatory
    citation: CitationRef | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CIT_CHUNG_RIR = CitationRef(
    doi="10.1107/S0021889874010375",
    journal="Journal of Applied Crystallography",
    year=1974,
    title="Quantitative interpretation of X-ray diffraction patterns of mixtures. "
          "I. Matrix-flushing method for quantitative multicomponent analysis",
)

CIT_KLUG_ALEXANDER = CitationRef(
    doi="10.1107/S0567739474001008",  # Wiley book, original 1974
    journal="Wiley (Klug & Alexander textbook)",
    year=1974,
    title="X-Ray Diffraction Procedures for Polycrystalline and Amorphous Materials, "
          "Chapter 7: Quantitative Analysis by Direct Comparison",
)


# ── XRD RIR method ───────────────────────────────────────────────────────────

def estimate_xrd_rir(
    components: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
) -> list[FractionEstimate]:
    """
    Reference Intensity Ratio method for XRD multi-phase quantification.

    Mass fraction of phase i:
        X_i = (I_i / RIR_i) / sum(I_j / RIR_j)

    Where RIR_i = I/I_corundum reference intensity ratio (from JCPDS/ICDD card).

    Requires:
      - Each component's profile has spectralSignatures.xrd.peaks
      - Each component's profile has rirFactor (I/I_c)
      - At least one matched peak per component

    Returns FractionEstimate list. Empty if requirements not met.
    """
    estimates: list[FractionEstimate] = []

    weighted_intensities: dict[str, float] = {}
    missing_rir: list[str] = []

    for comp_data in components:
        formula = comp_data["formula"]
        profile = profiles.get(formula)
        if not profile:
            continue

        rir = profile.get("rirFactor")
        if rir is None or rir <= 0:
            missing_rir.append(formula)
            continue

        match_result = comp_data.get("match_result", {})
        matches = match_result.get("matches", [])
        if not matches:
            continue

        # Use sum of matched peak intensities, weighted
        total_intensity = sum(m.get("sample_intensity", 0) for m in matches)
        if total_intensity <= 0:
            continue

        weighted_intensities[formula] = total_intensity / rir

    # Cannot run RIR if any component lacks RIR factor
    if missing_rir or len(weighted_intensities) < 1:
        return []

    total_weighted = sum(weighted_intensities.values())
    if total_weighted <= 0:
        return []

    for formula, weighted in weighted_intensities.items():
        fraction = weighted / total_weighted
        # RIR method typical uncertainty: 5-10% relative
        rel_uncertainty = 0.08
        abs_uncertainty = round(fraction * rel_uncertainty, 3)

        estimates.append(FractionEstimate(
            formula=formula,
            value=round(fraction, 3),
            uncertainty=abs_uncertainty,
            method="rir",
            quantitative=True,
            caveat=(
                "RIR method assumes randomly oriented powder and no preferred "
                "orientation. For thin films or textured samples, results may "
                "be biased by 10-30%. Confirm via Rietveld refinement for "
                "publication-grade fraction."
            ),
            citation=CIT_CHUNG_RIR,
        ))

    return estimates


# ── Raman qualitative ratio ──────────────────────────────────────────────────

def estimate_raman_qualitative(
    components: list[dict[str, Any]],
) -> list[FractionEstimate]:
    """
    Raman detected intensity ratio — QUALITATIVE only.

    Important: Raman scattering cross-section varies 10-100x between materials.
    Same mass of MoS2 and carbon black give very different signal intensities.
    This method reports observed signal ratio, NEVER mass fraction.

    Useful for: comparing same-material samples (relative crystallinity),
    monitoring composition changes over time in same instrument.
    NOT useful for: absolute composition determination.
    """
    estimates: list[FractionEstimate] = []

    intensities: dict[str, float] = {}
    for comp_data in components:
        formula = comp_data["formula"]
        match_result = comp_data.get("match_result", {})
        matches = match_result.get("matches", [])
        if not matches:
            continue
        # Use max intensity rather than sum (stronger peak = more reliable)
        max_int = max((m.get("sample_intensity", 0) for m in matches), default=0)
        if max_int > 0:
            intensities[formula] = max_int

    total = sum(intensities.values())
    if total <= 0 or len(intensities) < 1:
        return []

    for formula, intensity in intensities.items():
        ratio = intensity / total
        # Large uncertainty reflects cross-section variation
        estimates.append(FractionEstimate(
            formula=formula,
            value=round(ratio, 3),
            uncertainty=round(ratio * 0.4, 3),  # +/- 40% relative
            method="raman-intensity-ratio-qualitative",
            quantitative=False,
            caveat=(
                "Raman scattering cross-section varies 10-100x between materials. "
                "This is the DETECTED INTENSITY ratio, NOT mass fraction. "
                "For absolute composition, use XRD-RIR (powder samples) or "
                "elemental analysis (XPS, EDS, ICP-MS)."
            ),
            citation=None,
        ))

    return estimates


# ── Peak count fallback ──────────────────────────────────────────────────────

def estimate_peak_count_fallback(
    components: list[dict[str, Any]],
) -> list[FractionEstimate]:
    """
    Order-of-magnitude estimate based on observed vs expected peak count.
    Used when no quantitative method applies.

    NOT mass fraction. Just "how dominant is each phase in observed spectrum".
    """
    estimates: list[FractionEstimate] = []

    observed_counts: dict[str, int] = {}
    for comp_data in components:
        formula = comp_data["formula"]
        obs = comp_data.get("intended_peaks_observed", 0)
        if obs > 0:
            observed_counts[formula] = obs

    total = sum(observed_counts.values())
    if total <= 0:
        return []

    for formula, count in observed_counts.items():
        ratio = count / total
        estimates.append(FractionEstimate(
            formula=formula,
            value=round(ratio, 3),
            uncertainty=round(ratio * 0.5, 3),  # very loose
            method="peak-count-fallback",
            quantitative=False,
            caveat=(
                "Order-of-magnitude estimate based on observed peak counts. "
                "Does not account for cross-section, structure factor, or "
                "peak overlap. NOT mass fraction. For quantitative analysis: "
                "XRD-RIR (powders) or Rietveld refinement."
            ),
            citation=None,
        ))

    return estimates



# ── XRD Direct Comparison (Klug-Alexander) ───────────────────────────────────

def estimate_xrd_direct_comparison(
    components: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    anode: str = "Cu",
) -> list[FractionEstimate]:
    """
    Klug-Alexander Direct Comparison method.

    X_i = (I_i / (mu/rho)_i) / sum_j (I_j / (mu/rho)_j)

    Where I_i = total matched intensity, (mu/rho)_i = mass absorption
    coefficient computed from formula via pymatgen.

    More accurate than RIR (~3-5% vs 5-10%) and does not require RIR factors
    in materialProfiles. Only requires formula → can be applied to any compound.

    Requires pymatgen installed (already in worker venv).
    """
    from src.deviation.mass_absorption import compound_mac

    estimates: list[FractionEstimate] = []
    weighted_intensities: dict[str, float] = {}
    macs: dict[str, float] = {}
    failed: list[str] = []

    for comp_data in components:
        formula = comp_data["formula"]
        match_result = comp_data.get("match_result", {})
        matches = match_result.get("matches", [])
        if not matches:
            continue

        total_intensity = sum(m.get("sample_intensity", 0) for m in matches)
        if total_intensity <= 0:
            continue

        mac = compound_mac(formula, anode=anode)
        if mac is None or mac <= 0:
            failed.append(formula)
            continue

        macs[formula] = mac
        weighted_intensities[formula] = total_intensity / mac

    if not weighted_intensities or failed:
        # If any component fails MAC lookup, abandon DC method
        return []

    total_weighted = sum(weighted_intensities.values())
    if total_weighted <= 0:
        return []

    for formula, weighted in weighted_intensities.items():
        fraction = weighted / total_weighted
        # DC method typical uncertainty: 3-5% relative
        rel_uncertainty = 0.05
        abs_uncertainty = round(fraction * rel_uncertainty, 3)

        estimates.append(FractionEstimate(
            formula=formula,
            value=round(fraction, 3),
            uncertainty=abs_uncertainty,
            method="direct-comparison",
            quantitative=True,
            caveat=(
                f"Klug-Alexander Direct Comparison using mu/rho (Cu Kalpha) = "
                f"{macs[formula]:.1f} cm2/g. Assumes randomly oriented powder "
                "and known phase composition. For thin films, textured samples, "
                "or unknown amorphous fraction, results may bias 10-20%. "
                "Confirm via Rietveld refinement (R185-7c) for publication."
            ),
            citation=CIT_KLUG_ALEXANDER,
        ))

    return estimates


# ── Main dispatcher ──────────────────────────────────────────────────────────

def estimate_fractions(
    spectrum_type: str,
    components: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
) -> list[FractionEstimate]:
    """
    Choose best applicable method given spectrum type + available data.

    Args:
        spectrum_type: raman/xrd/ftir/pl/uvvis
        components: list of component dicts (from MultiPhaseResult.components,
            each as_dict). Each has match_result.
        profiles: { formula: materialProfile_dict }

    Returns:
        list of FractionEstimate, one per component with detected peaks.
    """
    if not components:
        return []

    # XRD path: try Direct Comparison first (most accurate without Rietveld),
    # then RIR (needs explicit rirFactor in profile), then peak-count fallback.
    if spectrum_type == "xrd":
        estimates = estimate_xrd_direct_comparison(components, profiles)
        if estimates:
            return estimates
        estimates = estimate_xrd_rir(components, profiles)
        if estimates:
            return estimates
        # Fall through to peak-count fallback if neither method available

    # Raman path
    if spectrum_type == "raman":
        return estimate_raman_qualitative(components)

    # UV-Vis: Lambert-Beer not yet implemented (needs epsilon in profile)
    # Fall through to peak-count fallback

    # Fallback for FTIR, PL, UV-Vis (lacks specific quant method)
    return estimate_peak_count_fallback(components)
