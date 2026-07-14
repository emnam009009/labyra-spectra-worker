"""Cross-Spectrum Inference Engine. @phase R185-8."""
from src.csie.aggregator import aggregate_evidence, run_csie
from src.csie.ambiguity import (
    AmbiguousObservation,
    CandidateCause,
    DiscriminationExperiment,
    handle_ambiguous,
)
from src.csie.pipeline import run_csie_for_sample
from src.csie.types import ConsistencyCheck, CSIEResult, PhaseEvidence

__all__ = [
    "AmbiguousObservation",
    "CSIEResult",
    "CandidateCause",
    "ConsistencyCheck",
    "DiscriminationExperiment",
    "PhaseEvidence",
    "aggregate_evidence",
    "handle_ambiguous",
    "run_csie",
    "run_csie_for_sample",
]
