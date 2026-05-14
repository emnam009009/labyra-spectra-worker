"""XRD candidate lookup orchestrator.

For each XRD spectrum:
  1. Extract formula from sample_label (fallback: skip citation).
  2. Query COD + MP in parallel (formula search).
  3. Fetch CIF for each candidate, simulate XRD pattern.
  4. Score each candidate against user peaks.
  5. Return top N ranked candidates with citations.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from src.citation.cod_client import (
    cod_entry_to_citation,
    fetch_cod_cif,
    search_cod_by_formula,
)
from src.citation.formula import extract_formula_from_label, normalize_formula
from src.citation.mp_client import (
    fetch_mp_structure,
    mp_entry_to_citation,
    mp_structure_to_cif,
    search_mp_by_formula,
)
from src.citation.peak_matcher import match_peaks
from src.citation.cache import get_cache
from src.citation.types import Candidate, Citation
from src.citation.xrd_simulator import simulate_powder_pattern

logger = logging.getLogger(__name__)

MAX_TOP_CANDIDATES = 5
COD_CANDIDATE_LIMIT = 8
MP_CANDIDATE_LIMIT = 4


def lookup_xrd_candidates(
    user_peaks: list[dict[str, Any]],
    sample_label: str | None = None,
    chemical_formula: str | None = None,
    filename: str | None = None,
) -> dict[str, Any]:
    """Main entry: find best material candidates for user XRD peaks.

    chemical_formula: if user provided explicit formula, use it.
    sample_label: fallback to regex extraction.

    Returns:
      {
        "formula_used": "WO3" | None,
        "lookup_attempted": bool,
        "candidates": [Candidate.to_dict(), ...] sorted by score desc,
        "errors": [...],
      }
    """
    result: dict[str, Any] = {
        "formula_used": None,
        "lookup_attempted": False,
        "candidates": [],
        "errors": [],
    }

    # Resolve formula
    formula = chemical_formula or extract_formula_from_label(sample_label or "")
    if not formula:
        logger.info("No formula resolvable from label=%r, skipping citation", sample_label)
        result["errors"].append("no_formula_resolvable")
        return result

    normalized = normalize_formula(formula)
    result["formula_used"] = normalized
    result["lookup_attempted"] = True

    # Parallel COD + MP search
    candidates: list[Candidate] = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        cod_future = executor.submit(_lookup_cod, normalized)
        mp_future = executor.submit(_lookup_mp, normalized)
        try:
            candidates.extend(cod_future.result(timeout=20))
        except Exception as exc:  # noqa: BLE001
            logger.warning("COD lookup error: %s", exc)
            result["errors"].append(f"cod_lookup_failed: {exc}")
        try:
            candidates.extend(mp_future.result(timeout=20))
        except Exception as exc:  # noqa: BLE001
            logger.warning("MP lookup error: %s", exc)
            result["errors"].append(f"mp_lookup_failed: {exc}")

    # Simulate + score in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {
            executor.submit(_simulate_and_score, candidate, user_peaks): candidate
            for candidate in candidates
        }
        scored: list[Candidate] = []
        for fut in as_completed(future_map):
            candidate = future_map[fut]
            try:
                scored_candidate = fut.result(timeout=15)
                if scored_candidate:
                    scored.append(scored_candidate)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Simulation failed for %s/%s: %s",
                               candidate.citation.source, candidate.citation.id, exc)

    # Rank by score, take top N
    scored.sort(key=lambda c: c.match_score, reverse=True)
    top = scored[:MAX_TOP_CANDIDATES]
    result["candidates"] = [c.to_dict() for c in top]

    if top:
        logger.info(
            "XRD citation lookup %s: %d total candidates, top score=%.3f",
            normalized, len(candidates), top[0].match_score
        )
    else:
        logger.info("XRD citation lookup %s: no candidates passed scoring", normalized)

    return result


def _lookup_cod(formula: str) -> list[Candidate]:
    """Search COD and return Candidate stubs (no CIF fetched yet)."""
    entries = search_cod_by_formula(formula)[:COD_CANDIDATE_LIMIT]
    candidates: list[Candidate] = []
    for entry in entries:
        try:
            candidates.append(_cod_entry_to_candidate(entry))
        except (KeyError, ValueError) as exc:
            logger.debug("Skipping COD entry %s: %s", entry.get("file"), exc)
    return candidates


def _cod_entry_to_candidate(entry: dict[str, Any]) -> Candidate:
    citation = cod_entry_to_citation(entry)
    return Candidate(
        citation=citation,
        formula=str(entry.get("formula", "")).strip(),
        space_group=str(entry.get("sg", "")).strip(),
        space_group_number=int(entry["sgNumber"]) if entry.get("sgNumber") and str(entry["sgNumber"]).isdigit() else None,
        crystal_system=None,
        lattice_a=_safe_float(entry.get("a")),
        lattice_b=_safe_float(entry.get("b")),
        lattice_c=_safe_float(entry.get("c")),
        lattice_alpha=_safe_float(entry.get("alpha")),
        lattice_beta=_safe_float(entry.get("beta")),
        lattice_gamma=_safe_float(entry.get("gamma")),
    )


def _lookup_mp(formula: str) -> list[Candidate]:
    """Search MP and return Candidate stubs (no structure fetched yet)."""
    entries = search_mp_by_formula(formula)[:MP_CANDIDATE_LIMIT]
    candidates: list[Candidate] = []
    for entry in entries:
        try:
            candidates.append(_mp_entry_to_candidate(entry))
        except (KeyError, ValueError) as exc:
            logger.debug("Skipping MP entry %s: %s", entry.get("material_id"), exc)
    return candidates


def _mp_entry_to_candidate(entry: dict[str, Any]) -> Candidate:
    citation = mp_entry_to_citation(entry)
    symmetry = entry.get("symmetry", {})
    return Candidate(
        citation=citation,
        formula=entry.get("formula_pretty", ""),
        space_group=symmetry.get("symbol", ""),
        space_group_number=symmetry.get("number"),
        crystal_system=symmetry.get("crystal_system"),
        lattice_a=None,
        lattice_b=None,
        lattice_c=None,
        lattice_alpha=None,
        lattice_beta=None,
        lattice_gamma=None,
    )


def _simulate_and_score(candidate: Candidate, user_peaks: list[dict[str, Any]]) -> Candidate | None:
    """Fetch CIF for candidate, simulate XRD, compute match score.

    Returns updated candidate or None if simulation failed.
    """
    cif_text: str | None = None
    source = candidate.citation.source

    if source == "COD":
        cif_text = fetch_cod_cif(candidate.citation.id)
    elif source == "MP":
        struct = fetch_mp_structure(candidate.citation.id)
        if struct:
            cif_text = mp_structure_to_cif(struct, material_id=candidate.citation.id)
            # Populate lattice from MP structure
            lattice = struct.get("lattice", {})
            candidate.lattice_a = lattice.get("a")
            candidate.lattice_b = lattice.get("b")
            candidate.lattice_c = lattice.get("c")
            candidate.lattice_alpha = lattice.get("alpha")
            candidate.lattice_beta = lattice.get("beta")
            candidate.lattice_gamma = lattice.get("gamma")

    if not cif_text:
        return None

    simulated = simulate_powder_pattern(cif_text)
    if not simulated:
        return None

    candidate.simulated_peaks = simulated[:20]  # cap for prompt
    match = match_peaks(user_peaks, simulated)
    candidate.match_score = match["score"]
    candidate.matched_peaks_count = match["matched_count"]
    candidate.total_user_peaks = match["total_user_peaks"]
    candidate.intensity_correlation = match["intensity_correlation"]

    return candidate


def _safe_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
