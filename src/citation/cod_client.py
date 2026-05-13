"""Crystallography Open Database client.

Uses public HTTP REST endpoint at http://www.crystallography.net/cod/
NOTE: HTTP only (server does not support HTTPS).

Search strategy: element params (el1, el2, ..., strictmin, strictmax)
Why not formula= : COD's formula param does substring match — unreliable.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from src.citation.formula import elements_only, parse_formula
from src.citation.types import Citation

logger = logging.getLogger(__name__)

COD_BASE_URL = "http://www.crystallography.net/cod"
COD_TIMEOUT_SECONDS = 15
COD_MAX_RESULTS = 15


def search_cod_by_formula(formula: str, *, max_results: int = COD_MAX_RESULTS) -> list[dict[str, Any]]:
    """Search COD entries by elements derived from formula.

    Returns list of dicts with metadata (file, sg, a/b/c/angles, formula, citation fields).
    Empty list on failure or no results.
    """
    elements = elements_only(formula)
    if not elements:
        return []
    if len(elements) > 5:
        # COD supports up to el1..el5
        elements = elements[:5]

    params: dict[str, str] = {
        "strictmin": str(len(elements)),
        "strictmax": str(len(elements)),
        "format": "json",
    }
    for i, el in enumerate(elements, start=1):
        params[f"el{i}"] = el

    try:
        response = requests.get(
            f"{COD_BASE_URL}/result",
            params=params,
            timeout=COD_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            logger.warning("COD returned non-list: %s", type(data).__name__)
            return []
        # Filter by formula stoichiometry match
        target = parse_formula(formula)
        target_ratios = _normalize_ratios(target)
        filtered = []
        for entry in data:
            entry_formula = entry.get("formula") or entry.get("calcformula") or ""
            entry_parsed = parse_formula(entry_formula.replace("-", "").strip())
            if not entry_parsed:
                continue
            entry_ratios = _normalize_ratios(entry_parsed)
            if _ratios_match(target_ratios, entry_ratios, tol=0.05):
                filtered.append(entry)
            if len(filtered) >= max_results:
                break
        logger.info("COD search %s: %d total → %d formula-matched", formula, len(data), len(filtered))
        return filtered
    except requests.RequestException as exc:
        logger.warning("COD search failed for %s: %s", formula, exc)
        return []
    except ValueError as exc:
        logger.warning("COD JSON parse failed for %s: %s", formula, exc)
        return []


def _normalize_ratios(parsed: dict[str, float]) -> dict[str, float]:
    if not parsed:
        return {}
    total = sum(parsed.values())
    return {k: v / total for k, v in parsed.items()}


def _ratios_match(a: dict[str, float], b: dict[str, float], tol: float = 0.05) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        if abs(a[k] - b[k]) > tol:
            return False
    return True


def fetch_cod_cif(cod_id: str) -> str | None:
    """Download CIF file for a COD entry."""
    try:
        response = requests.get(
            f"{COD_BASE_URL}/{cod_id}.cif",
            timeout=COD_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.text
    except requests.RequestException as exc:
        logger.warning("COD CIF fetch failed for %s: %s", cod_id, exc)
        return None


def cod_entry_to_citation(entry: dict[str, Any]) -> Citation:
    """Convert COD JSON entry to Citation."""
    cod_id = str(entry.get("file", ""))
    return Citation(
        source="COD",
        id=cod_id,
        authors=entry.get("authors"),
        title=entry.get("title"),
        journal=entry.get("journal"),
        year=int(entry["year"]) if entry.get("year") and str(entry["year"]).isdigit() else None,
        doi=entry.get("doi"),
        url=f"{COD_BASE_URL}/{cod_id}.html",
    )
