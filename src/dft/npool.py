"""Automatic ``-npool`` selection for pw.x.

The optimal k-point pool count depends on the number of *irreducible* k-points,
which is a function of the k-mesh **and** the crystal symmetry — so it differs
per material. This module computes the irreducible k-point count ahead of running
QE (via spglib through pymatgen's SpacegroupAnalyzer for automatic meshes; the
explicit point count for a bands path) and chooses npool to maximise *effective*
k-parallelism.

Trade-off encoded:
  - k-point parallelisation (-npool) has near-linear scaling and low
    communication, so we want npool as large as possible;
  - but npool ≤ n_k, npool must divide NPROC, and each pool needs enough ranks
    for the plane-wave/FFT parallelisation — the denser the charge grid
    (ecutrho, large for PAW/hard pseudopotentials) and the larger the cell, the
    higher that floor;
  - and an npool that splits n_k unevenly wastes cores in the busiest pool.

Score = npool × load_efficiency, where load_efficiency = n_k / (npool·⌈n_k/npool⌉)
∈ (0,1]. This maximises *effective* parallel k-work rather than nominal npool, so
a perfectly-balanced npool=1 never beats a slightly-uneven npool that actually
distributes the work (the failure mode of a hard "must divide evenly" rule).
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def min_ranks_per_pool(ecutrho_ry: float, n_atoms: int) -> int:
    """Floor on ranks per pool so the dense FFT grid distributes well. Scales with
    the charge-density cutoff (PAW/hard pseudo → large grids) and cell size."""
    base = 4
    if ecutrho_ry >= 600:
        base = 7  # PAW / hard pseudopotentials, dense FFT (e.g. 720 Ry)
    elif ecutrho_ry >= 400:
        base = 6
    if n_atoms >= 100:
        base = max(base, 8)  # large slabs / heterostructures
    return base


def pick_npool(nproc: int, n_k: int, *, ecutrho_ry: float = 400.0, n_atoms: int = 1) -> int:
    """Choose npool for ``nproc`` MPI ranks over ``n_k`` irreducible k-points."""
    if nproc <= 1 or n_k <= 1:
        return 1
    floor = min_ranks_per_pool(ecutrho_ry, n_atoms)
    cands = [d for d in _divisors(nproc) if d <= n_k and nproc // d >= floor]
    if not cands:
        # No divisor meets the FFT floor — relax it rather than forfeit all
        # k-parallelism (still keep ≥2 ranks/pool).
        cands = [d for d in _divisors(nproc) if d <= n_k and nproc // d >= 2] or [1]

    def score(d: int) -> float:
        per_max = math.ceil(n_k / d)
        load_eff = n_k / (d * per_max)  # ∈ (0,1]
        return d * load_eff

    cands.sort(key=lambda d: -score(d))
    return cands[0]


def irreducible_kpoints(structure: dict[str, Any], kpoints: dict[str, Any], nspin: int = 1) -> int | None:
    """Irreducible k-point count for a unit's K_POINTS setting, or None if it
    can't be determined (caller then falls back to npool=1).

    - ``automatic`` grid → spglib ``get_ir_reciprocal_mesh`` on the reconstructed
      cell (this is the count QE will print as "number of k points").
    - ``crystal_b`` path → the explicit number of path points (bands runs list
      every point; symmetry reduction does not apply).
    Spin-polarised (nspin=2) runs double the k-point work in QE's pool split.
    """
    if not kpoints:
        return None
    ktype = kpoints.get("type")
    try:
        if ktype == "crystal_b":
            path = kpoints.get("path") or []
            n = sum(int(p.get("npoints", 1)) for p in path) or len(path)
            return n * (2 if nspin == 2 else 1) or None
        if ktype == "automatic":
            grid = kpoints.get("grid")
            shift = kpoints.get("shift") or [0, 0, 0]
            if not grid or len(grid) != 3:
                return None
            from src.dft.scene import _reconstruct
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

            struct = _reconstruct(structure)
            sga = SpacegroupAnalyzer(struct, symprec=1e-3)
            is_shift = [1 if s else 0 for s in shift]
            ir = sga.get_ir_reciprocal_mesh(tuple(int(g) for g in grid), is_shift=tuple(is_shift))
            n = len(ir)
            return n * (2 if nspin == 2 else 1) or None
    except Exception as exc:  # noqa: BLE001 — auto-npool is best-effort; fall back to 1
        logger.warning("auto-npool: irreducible k-point count failed: %s", exc)
        return None
    return None


def auto_npool(
    structure: dict[str, Any],
    kpoints: dict[str, Any] | None,
    *,
    nproc: int,
    ecutrho_ry: float = 400.0,
    n_atoms: int = 1,
    nspin: int = 1,
) -> int:
    """End-to-end: irreducible k-points → npool. Returns 1 when k-points can't be
    resolved (safe: QE runs single-pool)."""
    if not kpoints:
        return 1
    n_k = irreducible_kpoints(structure, kpoints, nspin=nspin)
    if not n_k:
        return 1
    npool = pick_npool(nproc, n_k, ecutrho_ry=ecutrho_ry, n_atoms=n_atoms)
    logger.info(
        "auto-npool: n_k=%d nproc=%d ecutrho=%.0f atoms=%d → npool=%d (%d ranks/pool)",
        n_k, nproc, ecutrho_ry, n_atoms, npool, nproc // npool,
    )
    return npool
