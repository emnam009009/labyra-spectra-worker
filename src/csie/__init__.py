"""Cross-Spectrum Inference Engine. @phase R185-8."""
from src.csie.types import PhaseEvidence, ConsistencyCheck, CSIEResult
from src.csie.aggregator import aggregate_evidence, run_csie
from src.csie.pipeline import run_csie_for_sample
from src.csie.ambiguity import (
    handle_ambiguous,
    AmbiguousObservation,
    CandidateCause,
    DiscriminationExperiment,
)
