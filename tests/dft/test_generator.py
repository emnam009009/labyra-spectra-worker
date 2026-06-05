"""Tests for dft.generator — ordered QE pw.x input rendering.

@phase R272w-b (DFT P0)
"""
import pytest
from pymatgen.core import Lattice, Structure

from src.dft.generator import generate_pw_input, qe_bool, qe_sci
from src.dft.kpath import get_kpath
from src.dft.structure import to_dft_structure


def _rutile():
    return Structure.from_spacegroup(
        136, Lattice.tetragonal(4.594, 2.959), ["Ti", "O"], [[0, 0, 0], [0.305, 0.305, 0]]
    )


def _s(pseudo=True):
    return to_dft_structure(_rutile(), {"Ti": "Ti.UPF", "O": "O.UPF"} if pseudo else None)


def test_qe_filters():
    assert qe_bool(True) == ".true."
    assert qe_bool(False) == ".false."
    assert qe_sci(1e-9) == "1.0d-9"
    assert qe_sci(1e-10) == "1.0d-10"


def test_pw_scf_ordering_and_content():
    out = generate_pw_input(
        _s(),
        {
            "calculation": "scf", "tstress": False, "tprnfor": True,
            "occupations": "fixed", "convThr": 1e-9,
            "kPoints": {"type": "automatic", "grid": [6, 6, 8], "shift": [0, 0, 0]},
        },
        prefix="TiO2", ecutwfc=50, ecutrho=400,
        hubbard=[{"manifold": "Ti-3d", "value": 3.0}],
    )
    assert out.index("&CONTROL") < out.index("&SYSTEM") < out.index("&ELECTRONS")
    assert out.index("ATOMIC_SPECIES") < out.index("ATOMIC_POSITIONS") < out.index("K_POINTS")
    assert "ibrav       = 6" in out
    assert "celldm(1)   = 8.681402" in out  # 6-decimal format
    assert "conv_thr         = 1.0d-9" in out
    assert "K_POINTS {automatic}" in out and "6 6 8 0 0 0" in out
    assert "HUBBARD {ortho-atomic}" in out and "U Ti-3d 3.0" in out
    assert "occupations = 'fixed'" in out


def test_scf_omits_ions_cell_and_hubbard():
    out = generate_pw_input(
        _s(),
        {"calculation": "scf", "tstress": False, "tprnfor": True, "occupations": "fixed",
         "convThr": 1e-9, "kPoints": {"type": "automatic", "grid": [4, 4, 4], "shift": [0, 0, 0]}},
        prefix="x", ecutwfc=50, ecutrho=400,
    )
    assert "&IONS" not in out and "&CELL" not in out
    assert "HUBBARD" not in out


def test_vc_relax_includes_ions_cell():
    out = generate_pw_input(
        _s(),
        {"calculation": "vc-relax", "tstress": True, "tprnfor": True, "occupations": "smearing",
         "convThr": 1e-8, "kPoints": {"type": "automatic", "grid": [3, 3, 4], "shift": [0, 0, 0]}},
        prefix="x", ecutwfc=50, ecutrho=400,
    )
    assert "&IONS" in out and "&CELL" in out
    assert "cell_dofree    = 'all'" in out
    assert "tstress        = .true." in out
    assert "smearing" in out and "degauss" in out


def test_bands_crystal_b():
    rut = _rutile()
    out = generate_pw_input(
        to_dft_structure(rut, {"Ti": "Ti.UPF", "O": "O.UPF"}),
        {"calculation": "bands", "tstress": False, "tprnfor": False, "occupations": "fixed",
         "convThr": 1e-10, "kPoints": {"type": "crystal_b", "path": get_kpath(rut)["path"]}},
        prefix="x", ecutwfc=50, ecutrho=400,
    )
    assert "K_POINTS {crystal_b}" in out
    assert "! GAMMA" in out


def test_ibrav0_renders_cell_parameters():
    al = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.05), ["Al"], [[0, 0, 0]])
    s = to_dft_structure(al, {"Al": "Al.UPF"})
    assert s["ibrav"] == 0
    out = generate_pw_input(
        s,
        {"calculation": "scf", "tstress": False, "tprnfor": True, "occupations": "smearing",
         "convThr": 1e-8, "kPoints": {"type": "automatic", "grid": [12, 12, 12], "shift": [1, 1, 1]}},
        prefix="Al", ecutwfc=40, ecutrho=320,
    )
    assert "CELL_PARAMETERS angstrom" in out


def test_unsupported_calc_raises():
    with pytest.raises(ValueError, match="unsupported"):
        generate_pw_input(_s(), {"calculation": "dos", "kPoints": {}}, prefix="x", ecutwfc=1, ecutrho=1)
