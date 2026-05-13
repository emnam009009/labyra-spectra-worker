"""Materials Project REST API client (v0.86+).

Auth: X-API-KEY header. Key loaded from MP_API_KEY env or Secret Manager.

Strategy:
  - Query summary endpoint with formula filter.
  - Filter !theoretical AND energy_above_hull < 0.05 (stable phases only).
  - Fetch full structure for top hits → convert to CIF text for Dans_Diffraction.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

from src.citation.types import Citation

logger = logging.getLogger(__name__)

MP_BASE_URL = "https://api.materialsproject.org"
MP_TIMEOUT_SECONDS = 15
MP_MAX_RESULTS = 5
MP_STABILITY_THRESHOLD_EV = 0.05  # energy_above_hull cutoff


def _get_api_key() -> str | None:
    return os.environ.get("MP_API_KEY")


def search_mp_by_formula(formula: str, *, max_results: int = MP_MAX_RESULTS) -> list[dict[str, Any]]:
    """Search MP for stable, experimentally-relevant structures matching formula."""
    api_key = _get_api_key()
    if not api_key:
        logger.warning("MP_API_KEY not set, skipping MP search")
        return []

    fields = ",".join([
        "material_id",
        "formula_pretty",
        "symmetry",
        "density",
        "energy_above_hull",
        "theoretical",
        "database_IDs",
    ])
    params = {
        "formula": formula,
        "theoretical": "false",
        "energy_above_hull_max": str(MP_STABILITY_THRESHOLD_EV),
        "_fields": fields,
        "_limit": str(max_results),
    }

    try:
        response = requests.get(
            f"{MP_BASE_URL}/materials/summary/",
            params=params,
            headers={"X-API-KEY": api_key},
            timeout=MP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        logger.info("MP search %s: %d stable hits", formula, len(data))
        return data
    except requests.RequestException as exc:
        logger.warning("MP search failed for %s: %s", formula, exc)
        return []


def fetch_mp_structure(material_id: str) -> dict[str, Any] | None:
    """Fetch full pymatgen Structure dict for a material_id."""
    api_key = _get_api_key()
    if not api_key:
        return None
    try:
        response = requests.get(
            f"{MP_BASE_URL}/materials/summary/",
            params={
                "material_ids": material_id,
                "_fields": "material_id,structure",
            },
            headers={"X-API-KEY": api_key},
            timeout=MP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        if data and "structure" in data[0]:
            return data[0]["structure"]
    except requests.RequestException as exc:
        logger.warning("MP structure fetch failed for %s: %s", material_id, exc)
    return None


def mp_structure_to_cif(structure: dict[str, Any], material_id: str = "mp_struct") -> str:
    """Convert pymatgen Structure dict to minimal CIF text.

    NOTE: Uses P1 symmetry (no symmetry operations). For Dans_Diffraction
    powder simulation this is acceptable as it expands the unit cell explicitly.
    """
    lattice = structure["lattice"]
    sites = structure.get("sites", [])

    a = lattice["a"]
    b = lattice["b"]
    c = lattice["c"]
    alpha = lattice["alpha"]
    beta = lattice["beta"]
    gamma = lattice["gamma"]

    lines = [
        f"data_{material_id.replace('-', '_')}",
        f"_cell_length_a {a:.6f}",
        f"_cell_length_b {b:.6f}",
        f"_cell_length_c {c:.6f}",
        f"_cell_angle_alpha {alpha:.4f}",
        f"_cell_angle_beta {beta:.4f}",
        f"_cell_angle_gamma {gamma:.4f}",
        "_symmetry_space_group_name_H-M 'P 1'",
        "_symmetry_Int_Tables_number 1",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
    ]

    for i, site in enumerate(sites):
        species_list = site.get("species", [])
        if not species_list:
            continue
        primary = species_list[0]
        element = primary["element"]
        occupancy = primary.get("occu", 1.0)
        frac = site.get("abc", [0, 0, 0])
        label = f"{element}{i + 1}"
        lines.append(
            f"{label} {element} {frac[0]:.6f} {frac[1]:.6f} {frac[2]:.6f} {occupancy:.4f}"
        )

    return "\n".join(lines) + "\n"


def mp_entry_to_citation(entry: dict[str, Any]) -> Citation:
    """Convert MP summary entry to Citation."""
    mp_id = entry.get("material_id", "")
    icsd_ids = entry.get("database_IDs", {}).get("icsd", [])
    icsd_ref = icsd_ids[0] if icsd_ids else None
    return Citation(
        source="MP",
        id=mp_id,
        title=f"Materials Project entry {mp_id}" + (f" (ICSD {icsd_ref})" if icsd_ref else ""),
        journal="Materials Project Database",
        url=f"https://next-gen.materialsproject.org/materials/{mp_id}",
    )
