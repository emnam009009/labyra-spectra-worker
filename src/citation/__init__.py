"""XRD/Raman citation lookup (COD + Materials Project)."""

from src.citation.lookup import lookup_xrd_candidates
from src.citation.types import Candidate, Citation

__all__ = ["lookup_xrd_candidates", "Candidate", "Citation"]
