"""Smoke test FTIR parser."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.parsers.ftir import parse_ftir

FIXTURE = Path(__file__).parent / "fixtures" / "ftir_pet_sample.csv"


def _generate_ftir_fixture() -> None:
    if FIXTURE.exists() and FIXTURE.stat().st_size > 200:
        return
    x = np.arange(400.0, 4000.1, 2.0)
    # PET-like: C=O (1715), C-O (1240, 1095), aromatic C=C (1505), C-H bend (1340, 720)
    peaks = [(1715, 35, 15), (1505, 20, 12), (1340, 15, 10),
             (1240, 30, 15), (1095, 25, 18), (720, 25, 10), (3000, 10, 50)]
    transmittance = np.full_like(x, 95.0)
    for c, depth, w in peaks:
        transmittance -= depth * np.exp(-((x - c) ** 2) / (2 * w**2))
    np.random.seed(42)
    transmittance += np.random.normal(0, 0.3, size=len(x))

    with FIXTURE.open("w") as f:
        f.write("# Synthetic PET FTIR (%T)\n# wavenumber_cm-1,transmittance_%\n")
        for xi, yi in zip(x, transmittance):
            f.write(f"{xi:.1f},{yi:.2f}\n")


@pytest.fixture(autouse=True)
def _fixture_setup() -> None:
    _generate_ftir_fixture()


def test_parse_ftir_basic() -> None:
    result = parse_ftir(FIXTURE.read_text())
    assert result["spectrum_type"] == "ftir"
    assert result["y_mode"] == "transmittance"


def test_parse_ftir_functional_groups() -> None:
    result = parse_ftir(FIXTURE.read_text())
    groups = result["functional_groups"]
    # Should detect at least C=O (1680-1750)
    assert any(g["name"] == "C=O stretch" for g in groups)


def test_parse_ftir_peaks() -> None:
    result = parse_ftir(FIXTURE.read_text())
    assert len(result["peaks"]) >= 4
