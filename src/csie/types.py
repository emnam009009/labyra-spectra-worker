"""Type definitions for Cross-Spectrum Inference Engine.

@phase R185-8a-csie-evidence-aggregation
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SpectrumType = Literal["raman", "xrd", "ftir", "pl", "uvvis", "uvvis_drs", "tga"]
ConsistencyVerdict = Literal["confirmed", "partial", "missing", "conflict"]


@dataclass
class EvidenceItem:
    """Single piece of evidence from one spectrum for one phase."""
    spectrum_id: str
    spectrum_type: str
    technique_strength: float       # 0..1 weight (Raman/XRD strongest, etc.)
    match_quality: str              # "excellent" | "good" | "fair" | "poor" | "missing"
    intent_coverage: float          # 0..1 from MultiPhaseResult.ComponentMatch
    hypotheses_count: int           # number of rules that fired
    notable_findings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PhaseEvidence:
    """All cross-spectrum evidence for one declared phase."""
    formula: str
    role: str
    declared_in_sample: bool
    spectra_supporting: list[EvidenceItem] = field(default_factory=list)
    spectra_missing: list[str] = field(default_factory=list)
    spectra_conflicting: list[str] = field(default_factory=list)

    # Derived metrics
    consistency_score: float = 0.0  # 0..1 weighted by technique_strength
    verdict: ConsistencyVerdict = "missing"
    reasoning: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConsistencyCheck:
    """Result of cross-spectrum consistency analysis."""
    sample_id_hash: str             # Hashed sample ID for logging safety
    tenant_id_hash: str
    measurements_analyzed: int
    spectrum_types_present: list[str]

    declared_phases: list[PhaseEvidence] = field(default_factory=list)
    unexpected_observations: list[str] = field(default_factory=list)  # phases observed but NOT declared

    overall_coherence_score: float = 0.0
    conflicts_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CSIEResult:
    """Top-level output of CSIE pipeline."""
    schema_version: int = 1
    status: Literal["ok", "insufficient_data", "rate_limited", "failed"] = "ok"
    consistency: ConsistencyCheck | None = None
    notes: list[str] = field(default_factory=list)
    computed_at: str = ""           # ISO timestamp
    idempotency_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "consistency": self.consistency.to_dict() if self.consistency else None,
            "notes": self.notes,
            "computed_at": self.computed_at,
            "idempotency_key": self.idempotency_key,
        }
