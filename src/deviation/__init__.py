"""Deviation analysis package. @phase R185."""
from src.deviation.peak_matcher import match_peaks, MatchResult
from src.deviation.hypothesis import Hypothesis, RuleCitation
from src.deviation.rules import run_rules, ALL_RULES
from src.deviation.pipeline import run_deviation_analysis
from src.deviation.multi_phase import (
    match_multi_phase,
    ComponentDeclaration,
    MultiPhaseResult,
)
from src.deviation.crystallinity import (
    classify_crystallinity,
    adaptive_tolerance,
    CrystallinityResult,
    SizeEstimate,
)
from src.deviation.composite_rules import run_composite_rules, COMPOSITE_RULES
