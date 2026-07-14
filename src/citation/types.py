"""Citation data structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

SourceType = Literal["COD", "MP", "internal", "web", "unverified"]


@dataclass
class Citation:
    """Bibliographic reference for a material entry."""

    source: SourceType
    id: str  # COD ID or MP ID or library doc ID
    authors: str | None = None
    title: str | None = None
    journal: str | None = None
    year: int | None = None
    doi: str | None = None
    url: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Candidate:
    """Material candidate matched against user spectrum."""

    citation: Citation
    formula: str
    space_group: str
    space_group_number: int | None
    crystal_system: str | None
    lattice_a: float | None
    lattice_b: float | None
    lattice_c: float | None
    lattice_alpha: float | None
    lattice_beta: float | None
    lattice_gamma: float | None

    # Simulation results vs user peaks
    simulated_peaks: list[dict] = field(default_factory=list)
    match_score: float = 0.0  # 0..1
    matched_peaks_count: int = 0
    total_user_peaks: int = 0
    intensity_correlation: float | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["citation"] = self.citation.to_dict()
        return d
