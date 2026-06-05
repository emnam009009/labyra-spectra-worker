"""Tests for dft.kpath — seekpath → QE crystal_b path.

@phase R272w-b (DFT P0)
"""
from pymatgen.core import Lattice, Structure

from src.dft.kpath import get_kpath


def test_kpath_silicon():
    si = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])
    kp = get_kpath(si)
    assert kp["path"], "non-empty path"
    for pt in kp["path"]:
        assert "label" in pt and "npoints" in pt
        assert len(pt["coords"]) == 3
    assert kp["path"][-1]["npoints"] == 1  # final point closes the line
    assert "GAMMA" in kp["point_coords"]
    assert kp["segments"]  # seekpath segments present
