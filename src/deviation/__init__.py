"""Deviation analysis package. @phase R185."""
from src.deviation.composite_rules import COMPOSITE_RULES, run_composite_rules
from src.deviation.crystallinity import (
    CrystallinityResult,
    SizeEstimate,
    adaptive_tolerance,
    classify_crystallinity,
)
from src.deviation.fraction_estimator import (
    FractionEstimate,
    estimate_fractions,
    estimate_xrd_direct_comparison,
)
from src.deviation.hypothesis import Hypothesis, RuleCitation
from src.deviation.mass_absorption import compound_mac
from src.deviation.multi_phase import (
    ComponentDeclaration,
    MultiPhaseResult,
    match_multi_phase,
)
from src.deviation.peak_matcher import MatchResult, match_peaks
from src.deviation.pipeline import run_deviation_analysis
from src.deviation.rules import ALL_RULES, run_rules

__all__ = [
    "ALL_RULES",
    "COMPOSITE_RULES",
    "ComponentDeclaration",
    "CrystallinityResult",
    "FractionEstimate",
    "Hypothesis",
    "MatchResult",
    "MultiPhaseResult",
    "RuleCitation",
    "SizeEstimate",
    "adaptive_tolerance",
    "classify_crystallinity",
    "compound_mac",
    "estimate_fractions",
    "estimate_xrd_direct_comparison",
    "match_multi_phase",
    "match_peaks",
    "run_composite_rules",
    "run_deviation_analysis",
    "run_rules",
]
