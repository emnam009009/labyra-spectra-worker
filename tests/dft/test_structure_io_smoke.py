"""
Smoke / golden tests for dft.structure_io — VERIFY-not-TRUST ibrav mapping.

Builds canonical structures with pymatgen (no external files needed) and asserts
the ibrav decision + round-trip verification + sanity behaviour documented in
docs/scientific-methods/qe-lattice-ibrav.md. Mirrors nAM's ground-truth results
(hexagonal→ibrav4, fcc/centered→ibrav0, rutile→ibrav6).

@phase R272 (DFT P0 — worker scaffold)
"""
import pytest
from pymatgen.core import Lattice, Structure

from src.dft.structure_io import emit_qe, structure_to_qe


def test_simple_cubic_maps_to_ibrav_1():
    po = Structure.from_spacegroup("Pm-3m", Lattice.cubic(3.35), ["Po"], [[0, 0, 0]])
    r = structure_to_qe(po)
    assert r["crystal_system"] == "cubic"
    assert r["centering"] == "P"
    assert r["ibrav"] == 1
    assert r["verified"] is True
    assert r["sanity_errors"] == []


def test_fcc_centered_falls_back_to_ibrav_0():
    # Centered (F) lattice is NOT whitelisted → robust ibrav=0 (the §1 trap).
    al = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.05), ["Al"], [[0, 0, 0]])
    r = structure_to_qe(al)
    assert r["centering"] == "F"
    assert r["ibrav"] == 0  # never a wrong ibrav!=0 for centered cells


def test_hexagonal_maps_to_ibrav_4_verified():
    mg = Structure.from_spacegroup(
        194, Lattice.hexagonal(3.21, 5.21), ["Mg"], [[1 / 3, 2 / 3, 0.25]]
    )
    r = structure_to_qe(mg)
    assert r["crystal_system"] == "hexagonal"
    assert r["ibrav"] == 4
    assert r["verified"] is True


def test_rutile_tetragonal_maps_to_ibrav_6_verified():
    tio2 = Structure.from_spacegroup(
        136, Lattice.tetragonal(4.594, 2.959), ["Ti", "O"], [[0, 0, 0], [0.305, 0.305, 0]]
    )
    r = structure_to_qe(tio2)
    assert r["crystal_system"] == "tetragonal"
    assert r["ibrav"] == 6
    assert r["verified"] is True
    # celldm round-trips: a = 4.594 Å / 0.529177 ≈ 8.681 Bohr; c/a = 2.959/4.594
    assert abs(r["celldm"][1] - 8.6814) < 1e-3
    assert abs(r["celldm"][3] - 0.6441) < 1e-3


def test_emit_qe_renders_ibrav_block():
    tio2 = Structure.from_spacegroup(
        136, Lattice.tetragonal(4.594, 2.959), ["Ti", "O"], [[0, 0, 0], [0.305, 0.305, 0]]
    )
    sysl, body = emit_qe(structure_to_qe(tio2))
    assert "ibrav = 6" in sysl
    assert "celldm(1)" in sysl and "celldm(3)" in sysl
    assert "ATOMIC_POSITIONS crystal" in body


def test_sanity_overlap_raises():
    # Two atoms 0.05 Å apart → sanity must refuse to emit a broken input.
    bad = Structure(Lattice.cubic(5.0), ["H", "H"], [[0, 0, 0], [0.01, 0, 0]])
    r = structure_to_qe(bad, prefer_ibrav=False)
    assert r["sanity_errors"]  # non-empty
    with pytest.raises(ValueError, match="SANITY FAIL"):
        emit_qe(r)
