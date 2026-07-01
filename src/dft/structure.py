"""
structure.py — build the JSON DftStructure (the app's `types/dft.ts` §5.1 shape)
from a CIF / POSCAR string or a Materials Project id, reusing the verified ibrav
logic in structure_io.py. The worker emits exactly what the app stores and the
Jinja2 templates render. Sanity failures RAISE — never produce a broken structure.

`celldm` carries the verified values for whitelisted ibrav≠0; for ibrav=0 the cell
travels in `cellParameters` (3×3 Å) — the app/template renders CELL_PARAMETERS.

@phase R272w-b (DFT P0 — structure source)
"""
from __future__ import annotations

import logging
from typing import Any

from pymatgen.core import Element, Structure

from src.dft.structure_io import structure_to_qe

logger = logging.getLogger(__name__)


def _ordered_species(struct: Structure) -> list[str]:
    """Distinct element symbols in order of first appearance (QE ATOMIC_SPECIES)."""
    seen: list[str] = []
    for site in struct:
        sym = site.specie.symbol
        if sym not in seen:
            seen.append(sym)
    return seen


def to_dft_structure(
    struct: Structure,
    pseudo_map: dict[str, str] | None = None,
    use_primitive: bool = True,
    prefer_ibrav: bool = True,
) -> dict[str, Any]:
    """pymatgen Structure → DftStructure dict (matches app types/dft.ts §5.1).

    pseudo_map: element symbol → UPF filename, e.g. {"W": "W.pbe-...UPF"}. Missing
    elements get an empty pseudoFile (user fills from the library).
    Raises ValueError on sanity failure (overlapping atoms / degenerate cell).
    """
    pseudo_map = pseudo_map or {}
    res = structure_to_qe(struct, use_primitive=use_primitive, prefer_ibrav=prefer_ibrav)
    if res["sanity_errors"]:
        raise ValueError("SANITY FAIL: " + "; ".join(res["sanity_errors"]))

    s = res["structure"]
    species = [
        {"element": el, "mass": float(Element(el).atomic_mass), "pseudoFile": pseudo_map.get(el, "")}
        for el in _ordered_species(s)
    ]
    positions = [
        {
            "element": site.specie.symbol,
            "x": float(site.frac_coords[0]),
            "y": float(site.frac_coords[1]),
            "z": float(site.frac_coords[2]),
        }
        for site in s
    ]
    ibrav = res["ibrav"]
    return {
        "ibrav": ibrav,
        # JSON object keys stringify on serialization; Jinja2 .items() handles either.
        "celldm": dict(res["celldm"]) if res["celldm"] else {},
        "cellParameters": res["cell_ang"].tolist() if ibrav == 0 else None,
        # cellAng: FLAT 9 numbers (row-major) of the Angstrom cell, any ibrav —
        # used to reconstruct the structure for the 3D viewer / export. Flat (not
        # 3×3) because Firestore rejects nested arrays; scene.py reshapes it.
        "cellAng": res["cell_ang"].flatten().tolist(),
        "nat": len(s),
        "ntyp": len(species),
        "atomicSpecies": species,
        "atomicPositions": positions,
        "positionsType": "crystal",
        "spaceGroup": res["space_group"],
        "verified": res["verified"],
        "note": res["note"],
    }


def from_cif(cif_text: str, pseudo_map: dict[str, str] | None = None, **kw: Any) -> dict[str, Any]:
    """CIF text → DftStructure dict."""
    return to_dft_structure(Structure.from_str(cif_text, fmt="cif"), pseudo_map, **kw)


def from_poscar(poscar_text: str, pseudo_map: dict[str, str] | None = None, **kw: Any) -> dict[str, Any]:
    """POSCAR text → DftStructure dict."""
    return to_dft_structure(Structure.from_str(poscar_text, fmt="poscar"), pseudo_map, **kw)


def from_mp_id(
    mp_id: str, api_key: str, pseudo_map: dict[str, str] | None = None, **kw: Any
) -> dict[str, Any]:
    """Materials Project id → DftStructure dict (per-user MP key). CC-BY: record mp_id."""
    from mp_api.client import MPRester

    with MPRester(api_key) as mpr:
        struct = mpr.get_structure_by_material_id(mp_id)
    out = to_dft_structure(struct, pseudo_map, **kw)
    out["source"] = {"materials_project": True, "mp_id": mp_id}
    return out
