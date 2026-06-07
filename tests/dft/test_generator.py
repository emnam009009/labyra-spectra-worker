"""Tests for dft.generator — ordered QE pw.x input rendering.

@phase R272w-b (DFT P0)
"""
import pytest
from pymatgen.core import Lattice, Structure

from src.dft.generator import generate_postproc_input, generate_pw_input, qe_bool, qe_sci
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


# ── post-processing executables (bands.x / dos.x / projwfc.x / pp.x) ──────────


def test_postproc_bands():
    out = generate_postproc_input("ppbands", prefix="TiO2", functional="pbe", outdir="./out")
    assert "&BANDS" in out
    assert "PBE_TiO2.band" in out
    assert "lsym        = .true." in out


def test_postproc_dos_defaults():
    out = generate_postproc_input("dos", prefix="TiO2", functional="pbe")
    assert "&DOS" in out
    assert "PBE_TiO2.dos" in out
    assert "Emin    = -5.0" in out and "Emax    = 13.0" in out  # template defaults
    assert "ngauss  = -1" in out


def test_postproc_pdos_overrides():
    out = generate_postproc_input("pdos", {"emin": -8.0, "emax": 10.0}, prefix="WO3", functional="pbe")
    assert "&PROJWFC" in out
    assert "PBE_WO3.pdos" in out
    assert "Emin    = -8.0" in out and "Emax    = 10.0" in out


def test_postproc_charge_stm_sample_bias():
    out = generate_postproc_input(
        "charge", {"plotNum": 5, "sampleBias": 0.1}, prefix="WO3", name="stm", functional="hse"
    )
    assert "&INPUTPP" in out and "&PLOT" in out
    assert "plot_num    = 5" in out
    assert "sample_bias = +0.1" in out
    assert "charge/stm_WO3.cube" in out


def test_postproc_charge_no_sample_bias_when_not_5():
    out = generate_postproc_input("charge", {"plotNum": 0}, prefix="WO3", name="rho")
    assert "plot_num    = 0" in out
    assert "sample_bias" not in out
    assert "charge/rho_WO3.charge" in out


def test_postproc_unsupported_raises():
    with pytest.raises(ValueError, match="unsupported post-processing"):
        generate_postproc_input("scf", prefix="x")  # scf is pw.x, not post-proc


def test_occupations_omitted_when_absent():
    """bands has no occupations → must NOT render `occupations = ''` (QE: set_occupations error)."""
    # bands: no occupations key
    bands = generate_pw_input(
        _s(),
        {"calculation": "bands", "convThr": 1e-8,
         "kPoints": {"type": "crystal_b", "path": [
             {"coords": [0, 0, 0], "label": "G", "npoints": 20},
             {"coords": [0.5, 0, 0.5], "label": "X", "npoints": 1}]}},
        prefix="Si", ecutwfc=40, ecutrho=160,
    )
    assert "occupations" not in bands
    assert "occupations = ''" not in bands  # the exact bug
    # scf with smearing: occupations + smearing + degauss all present
    scf = generate_pw_input(
        _s(),
        {"calculation": "scf", "occupations": "smearing", "degauss": 0.01, "convThr": 1e-9,
         "kPoints": {"type": "automatic", "grid": [6, 6, 6], "shift": [0, 0, 0]}},
        prefix="Si", ecutwfc=40, ecutrho=160,
    )
    assert "occupations = 'smearing'" in scf
    assert "degauss" in scf


def test_vdw_d3_emitted_when_param_set():
    """vdW D3 (grimme-d3) for layered materials like 2H-WS2: emit vdw_corr +
    dftd3_version + dftd3_threebody only when params present; absent otherwise."""
    struct = {
        "ibrav": 4, "celldm": {1: 6.0296188, 3: 4.4511446}, "nat": 6, "ntyp": 2,
        "atomicSpecies": [
            {"element": "W", "mass": 183.84, "pseudoFile": "W.upf"},
            {"element": "S", "mass": 32.06, "pseudoFile": "S.upf"},
        ],
        "atomicPositions": [{"element": "W", "x": 0.0, "y": 0.0, "z": 0.0}],
        "positionsType": "crystal",
    }
    params = {
        "calculation": "scf", "occupations": "fixed", "convThr": 1e-9,
        "vdwCorr": "grimme-d3", "dftd3Version": 3, "dftd3Threebody": True,
        "kPoints": {"type": "automatic", "grid": [15, 15, 4], "shift": [0, 0, 0]},
    }
    out = generate_pw_input(struct, params, prefix="2H-WS2_bulk",
                            ecutwfc=60.0, ecutrho=720.0, functional="pbe",
                            hubbard=[{"manifold": "W-5d", "value": 6.2}], outdir="./out")
    assert "vdw_corr    = 'grimme-d3'" in out
    assert "dftd3_version = 3" in out
    assert "dftd3_threebody = .true." in out

    # without vdW params → no vdw_corr line
    params_no_vdw = {k: v for k, v in params.items() if not k.startswith(("vdw", "dftd3"))}
    out2 = generate_pw_input(struct, params_no_vdw, prefix="x", ecutwfc=60.0,
                             ecutrho=720.0, functional="pbe", hubbard=None, outdir="./out")
    assert "vdw_corr" not in out2
