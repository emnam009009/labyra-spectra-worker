"""
Hypothesis dataclass + Citation type for physics rules engine.

@phase R185-2-physics-rules-engine
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal


@dataclass
class RuleCitation:
    """DOI-backed citation for a physics rule."""
    doi: str
    journal: str
    year: int
    title: str
    verified: bool = True


@dataclass
class Hypothesis:
    """A physics-based hypothesis explaining observed deviation."""
    rule_id: str                                  # e.g. "R1-tensile-strain"
    name: str                                     # display label
    confidence: float                             # 0.0-1.0
    evidence: list[str] = field(default_factory=list)
    quantitative_estimate: str | None = None      # e.g. "strain ~ 0.8%"
    suggested_followup: str | None = None         # e.g. "confirm via XRD (002) shift"
    citation: RuleCitation | None = None
    severity: Literal["info", "notice", "warning", "critical"] = "info"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
