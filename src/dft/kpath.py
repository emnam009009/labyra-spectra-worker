"""
kpath.py — high-symmetry Brillouin-zone path via seekpath, formatted for a QE
`K_POINTS {crystal_b}` bands input. seekpath standardizes the cell, so a structure
whose bands use this path MUST be the standardized primitive cell (see
docs/scientific-methods/qe-lattice-ibrav.md §6). Experts may supply their own path.

@phase R272w-b (DFT P0 — k-path)
"""
from __future__ import annotations

import logging
from typing import Any

import seekpath
from pymatgen.core import Structure

logger = logging.getLogger(__name__)


def get_kpath(struct: Structure, npoints_per_segment: int = 20) -> dict[str, Any]:
    """pymatgen Structure → BZ path.

    Returns:
      point_coords : {label: [kx, ky, kz]} reciprocal-crystal coords
      segments     : [(labelA, labelB), ...] from seekpath
      path         : [{label, coords, npoints}, ...] ready for K_POINTS {crystal_b}.
                     `npoints` is the number of points to the NEXT line point; a
                     break (or the final point) carries 1 so QE does not interpolate
                     across a discontinuity.
    """
    cell = (
        struct.lattice.matrix.tolist(),
        [site.frac_coords.tolist() for site in struct],
        [site.specie.Z for site in struct],
    )
    res = seekpath.get_path(cell)
    coords: dict[str, list[float]] = {k: list(v) for k, v in res["point_coords"].items()}
    segments: list[tuple[str, str]] = [tuple(seg) for seg in res["path"]]

    path: list[dict[str, Any]] = []
    for i, (a, b) in enumerate(segments):
        contiguous = i > 0 and segments[i - 1][1] == a
        if not contiguous:
            if path:
                path[-1]["npoints"] = 1  # close the previous line at a break
            path.append({"label": a, "coords": coords[a], "npoints": npoints_per_segment})
        path.append({"label": b, "coords": coords[b], "npoints": npoints_per_segment})
    if path:
        path[-1]["npoints"] = 1  # final point: no continuation

    bravais = res.get("bravais_lattice_extended") or res.get("bravais_lattice")
    return {"point_coords": coords, "segments": segments, "path": path, "bravais": bravais}
