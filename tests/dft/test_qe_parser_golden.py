"""
Golden test for dft.qe_parser against nAM's real QE 7.4.x .out files.

DROP the 19 ground-truth .out files into tests/dft/fixtures/ to activate
(see fixtures/README.md). Without them the module skips — it never fails CI for
missing data. Documented expected values (from the DFT master spec) to pin once
fixtures are present:
  - parse_final_structure     → 12 atoms, volume ≈ 178.6 Å³
  - parse_scf_summary (PBE)    → gap ≈ 2.72 eV, JOB DONE, ~49 SCF iters
  - parse_bands                → 422 k-points, 100 bands
  - band_gap_from_eigenvalues  → PBE0 gap ≈ 1.16 eV

@phase R272 (DFT P0 — worker scaffold)
"""
import glob
import os

import pytest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
_out_files = sorted(glob.glob(os.path.join(FIXTURES, "*.out")))

pytestmark = pytest.mark.skipif(
    not _out_files, reason="no QE .out fixtures in tests/dft/fixtures/ (see README)"
)


def test_fixtures_parse_without_raising():
    """Activation check: every fixture parses. Pin exact goldens per-file later."""
    from src.dft.qe_parser import parse_scf_summary

    for path in _out_files:
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        summary = parse_scf_summary(text)
        assert summary is not None
    # TODO(nAM): per-file golden asserts (12 atoms / gap 2.72 / 422 k / gap 1.16).
