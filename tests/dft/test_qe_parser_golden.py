"""
Golden tests for dft.qe_parser — two layers:

  1. SYNTHETIC (always runs): format-correct QE 7.4.x snippets committed as
     fixtures (synthetic_*.out). Exercises every parser with NO large files.
  2. REAL (optional): drop your actual .out files (any name except synthetic_*)
     into tests/dft/fixtures/ — invariant checks run automatically. Pin exact
     values in an optional fixtures/expected.json:
       {"my_scf.out": {"band_gap_ev": 2.72, "scf_iterations": 49},
        "my_relax.out": {"n_atoms": 12}, "my_bands.out": {"nks": 422}}
     (.out files there are gitignored — see fixtures/README.md.)

@phase R272w-d (DFT P0 — parser golden)
"""
import glob
import json
import os

import pytest

from src.dft.qe_parser import (
    band_gap_from_eigenvalues,
    parse_bands,
    parse_convergence,
    parse_final_structure,
    parse_scf_summary,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _read(name: str) -> str:
    with open(os.path.join(FIXTURES, name), encoding="utf-8", errors="ignore") as f:
        return f.read()


# ── Layer 1: synthetic golden (always runs, format-correct, tiny) ────────────


def test_synthetic_scf_summary():
    s = parse_scf_summary(_read("synthetic_scf.out"))
    assert s["band_gap_ev"] == 2.72  # 5.2200 - 2.5000
    assert s["total_energy_ry"] == pytest.approx(-500.12345678)
    assert s["scf_iterations"] == 42
    assert s["n_electrons"] == 64.0
    assert s["nbnd"] == 40
    assert s["alat_bohr"] == pytest.approx(8.6814)
    assert s["job_done"] is True


def test_synthetic_convergence():
    c = parse_convergence(_read("synthetic_relax.out"))
    assert c["n_ionic_steps"] == 2
    assert c["converged"] is True
    assert c["bfgs_steps"] == 1
    assert c["final_force"] == pytest.approx(0.0003)
    assert c["final_scf_accuracy"] == pytest.approx(2e-10)


def test_synthetic_final_structure():
    fs = parse_final_structure(_read("synthetic_relax.out"))
    assert fs["n_atoms"] == 3
    assert fs["volume_ang3"] == pytest.approx(178.60)
    assert fs["species"] == ["Ti", "Ti", "O"]
    assert fs["cell_ang"][0][0] == pytest.approx(4.594, abs=1e-2)


def test_synthetic_bands_and_gap():
    b = parse_bands(_read("synthetic_bands.out"))
    assert b["nks_declared"] == 4 and b["nks_parsed"] == 4
    assert b["nbnd"] == 6
    g = band_gap_from_eigenvalues(b, n_electrons=8)
    assert g["band_gap_ev"] == pytest.approx(8.9)
    assert g["vbm_ev"] == pytest.approx(-2.4)
    assert g["cbm_ev"] == pytest.approx(6.5)


# ── Layer 2: real .out files (optional — invariants + optional expected.json) ─

_real = [
    p for p in sorted(glob.glob(os.path.join(FIXTURES, "*.out")))
    if not os.path.basename(p).startswith("synthetic_")
]


def _expected() -> dict:
    path = os.path.join(FIXTURES, "expected.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


@pytest.mark.skipif(
    not _real, reason="no real .out fixtures (drop yours in; synthetic golden still runs)"
)
@pytest.mark.parametrize("path", _real, ids=lambda p: os.path.basename(p))
def test_real_out_invariants_and_expected(path):
    name = os.path.basename(path)
    text = _read(name)
    exp = _expected().get(name, {})

    summary = parse_scf_summary(text)
    if summary["band_gap_ev"] is not None:
        assert summary["band_gap_ev"] >= 0, "band gap must be non-negative"
    if summary["total_energy_ry"] is not None:
        assert summary["total_energy_ry"] < 0, "QE total energy is negative"

    bands = parse_bands(text)
    if bands["eigenvalues"]:
        assert len({len(e) for e in bands["eigenvalues"]}) == 1, "consistent band count per k"

    struct = parse_final_structure(text)
    if struct is not None:
        assert struct["n_atoms"] >= 1
        if struct["volume_ang3"] is not None:
            assert struct["volume_ang3"] > 0

    # exact-value pinning (only if provided in expected.json)
    if "band_gap_ev" in exp:
        assert summary["band_gap_ev"] == pytest.approx(exp["band_gap_ev"], abs=0.01)
    if "scf_iterations" in exp:
        assert summary["scf_iterations"] == exp["scf_iterations"]
    if "n_atoms" in exp and struct is not None:
        assert struct["n_atoms"] == exp["n_atoms"]
    if "nks" in exp:
        assert bands["nks_parsed"] == exp["nks"]
