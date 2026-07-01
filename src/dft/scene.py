"""
Structure scene + export for the crystal viewer (R327 Phase 1).

Reconstructs a pymatgen Structure from a stored DftStructure (fractional coords +
Angstrom cell) and produces:
  - build_scene:  a lightweight JSON scene (atoms + CrystalNN bonds, Cartesian Å,
    Jmol colours + atomic radii) for the app's Three.js renderer.
  - export_structure:  CIF / POSCAR text via pymatgen.

Uses the same bonding algorithm (CrystalNN) as the Materials Project; rendering is
done client-side so we ship no heavy Python viewer deps.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Jmol / CPK element colours (hex) — common elements, purple fallback for the rest.
_JMOL_COLORS: dict[str, str] = {
    "H": "#FFFFFF", "He": "#D9FFFF", "Li": "#CC80FF", "Be": "#C2FF00",
    "B": "#FFB5B5", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D",
    "F": "#90E050", "Ne": "#B3E3F5", "Na": "#AB5CF2", "Mg": "#8AFF00",
    "Al": "#BFA6A6", "Si": "#F0C8A0", "P": "#FF8000", "S": "#FFFF30",
    "Cl": "#1FF01F", "Ar": "#80D1E3", "K": "#8F40D4", "Ca": "#3DFF00",
    "Sc": "#E6E6E6", "Ti": "#BFC2C7", "V": "#A6A6AB", "Cr": "#8A99C7",
    "Mn": "#9C7AC7", "Fe": "#E06633", "Co": "#F090A0", "Ni": "#50D050",
    "Cu": "#C88033", "Zn": "#7D80B0", "Ga": "#C28F8F", "Ge": "#668F8F",
    "As": "#BD80E3", "Se": "#FFA100", "Br": "#A62929", "Rb": "#702EB0",
    "Sr": "#00FF00", "Y": "#94FFFF", "Zr": "#94E0E0", "Nb": "#73C2C9",
    "Mo": "#54B5B5", "Tc": "#3B9E9E", "Ru": "#248F8F", "Rh": "#0A7D8C",
    "Pd": "#006985", "Ag": "#C0C0C0", "Cd": "#FFD98F", "In": "#A67573",
    "Sn": "#668080", "Sb": "#9E63B5", "Te": "#D47A00", "I": "#940094",
    "Cs": "#57178F", "Ba": "#00C900", "La": "#70D4FF", "Ce": "#FFFFC7",
    "Hf": "#4DC2FF", "Ta": "#4DA6FF", "W": "#2194D6", "Re": "#267DAB",
    "Os": "#266696", "Ir": "#175487", "Pt": "#D0D0E0", "Au": "#FFD123",
    "Hg": "#B8B8D0", "Tl": "#A6544D", "Pb": "#575961", "Bi": "#9E4FB5",
    "Th": "#00BAFF", "U": "#008FFF",
}


def _color(el: str) -> str:
    return _JMOL_COLORS.get(el, "#DDA0DD")


def _radius(el: str) -> float:
    from pymatgen.core import Element  # type: ignore[import]

    try:
        r = Element(el).atomic_radius
        return float(r) if r else 1.0
    except Exception:
        return 1.0


def _reconstruct(structure: dict[str, Any]):
    """DftStructure dict → pymatgen Structure (fractional coords, Å cell)."""
    from pymatgen.core import Lattice, Structure  # type: ignore[import]

    cell = structure.get("cellAng") or structure.get("cellParameters")
    if not cell:
        raise ValueError("structure has no Angstrom cell — re-import to enable 3D view")

    positions = structure.get("atomicPositions") or []
    species = [p["element"] for p in positions]
    coords = [[float(p["x"]), float(p["y"]), float(p["z"])] for p in positions]
    lattice = Lattice(cell)
    cartesian = structure.get("positionsType") == "angstrom"
    return Structure(lattice, species, coords, coords_are_cartesian=cartesian)


def build_scene(structure: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct + build a render scene: atoms (+ bonded image atoms) and bonds."""
    from pymatgen.analysis.graphs import StructureGraph  # type: ignore[import]
    from pymatgen.analysis.local_env import CrystalNN  # type: ignore[import]

    struct = _reconstruct(structure)
    lattice = struct.lattice

    atoms: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float]] = set()

    def add_atom(el: str, cart: Any) -> None:
        key = (round(float(cart[0]), 3), round(float(cart[1]), 3), round(float(cart[2]), 3))
        if key in seen:
            return
        seen.add(key)
        atoms.append(
            {
                "el": el,
                "xyz": [float(cart[0]), float(cart[1]), float(cart[2])],
                "color": _color(el),
                "radius": _radius(el),
            }
        )

    for site in struct:
        add_atom(site.specie.symbol, site.coords)

    bonds: list[dict[str, Any]] = []
    try:
        graph = StructureGraph.with_local_env_strategy(struct, CrystalNN())
        for i, j, data in graph.graph.edges(data=True):
            jimage = np.array(data.get("to_jimage", (0, 0, 0)))
            from_c = struct[i].coords
            to_c = lattice.get_cartesian_coords(struct[j].frac_coords + jimage)
            bonds.append(
                {
                    "from": [float(c) for c in from_c],
                    "to": [float(c) for c in to_c],
                }
            )
            if jimage.any():
                add_atom(struct[j].specie.symbol, to_c)
    except Exception:
        logger.exception("bond detection failed; returning atoms only")

    return {
        "formula": struct.composition.reduced_formula,
        "lattice": [[float(x) for x in row] for row in lattice.matrix.tolist()],
        "natoms": len(struct),
        "atoms": atoms,
        "bonds": bonds,
    }


def export_structure(structure: dict[str, Any], fmt: str) -> str:
    """Reconstruct + emit CIF / POSCAR text."""
    struct = _reconstruct(structure)
    pmg_fmt = {"cif": "cif", "poscar": "poscar"}.get(fmt.lower())
    if pmg_fmt is None:
        raise ValueError(f"unsupported export format: {fmt}")
    return str(struct.to(fmt=pmg_fmt))
