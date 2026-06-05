"""Tests for dft.structure — DftStructure dict from pymatgen, reusing structure_io.

@phase R272w-b (DFT P0)
"""
import pytest
from pymatgen.core import Lattice, Structure

from src.dft.structure import from_cif, to_dft_structure


def _rutile():
    return Structure.from_spacegroup(
        136, Lattice.tetragonal(4.594, 2.959), ["Ti", "O"], [[0, 0, 0], [0.305, 0.305, 0]]
    )


def test_to_dft_structure_rutile():
    s = to_dft_structure(_rutile(), {"Ti": "Ti.UPF", "O": "O.UPF"})
    assert s["ibrav"] == 6
    assert s["nat"] == 6 and s["ntyp"] == 2
    assert {d["element"] for d in s["atomicSpecies"]} == {"Ti", "O"}
    assert all(d["pseudoFile"] for d in s["atomicSpecies"])
    assert all(d["mass"] > 0 for d in s["atomicSpecies"])
    assert len(s["atomicPositions"]) == 6
    assert s["positionsType"] == "crystal"
    assert s["cellParameters"] is None  # ibrav != 0
    assert 1 in s["celldm"] and 3 in s["celldm"]
    assert s["verified"] is True


def test_missing_pseudo_is_empty():
    s = to_dft_structure(_rutile())  # no pseudo_map
    assert all(d["pseudoFile"] == "" for d in s["atomicSpecies"])


def test_fcc_ibrav0_has_cell_parameters():
    al = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.05), ["Al"], [[0, 0, 0]])
    s = to_dft_structure(al, {"Al": "Al.UPF"})
    assert s["ibrav"] == 0
    assert s["cellParameters"] is not None
    assert len(s["cellParameters"]) == 3 and len(s["cellParameters"][0]) == 3
    assert s["celldm"] == {}


def test_from_cif_roundtrip():
    cif = _rutile().to(fmt="cif")
    s = from_cif(cif, {"Ti": "Ti.UPF", "O": "O.UPF"})
    assert s["ntyp"] == 2 and s["nat"] >= 2


def test_sanity_overlap_raises():
    bad = Structure(Lattice.cubic(5.0), ["H", "H"], [[0, 0, 0], [0.01, 0, 0]])
    with pytest.raises(ValueError, match="SANITY FAIL"):
        to_dft_structure(bad, prefer_ibrav=False)
