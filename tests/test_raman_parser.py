"""Smoke test Raman parser."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.parsers.raman import parse_raman

FIXTURE = Path(__file__).parent / "fixtures" / "raman_graphene_sample.txt"


def _generate_raman_fixture() -> None:
    if FIXTURE.exists() and FIXTURE.stat().st_size > 200:
        return
    x = np.arange(100.0, 3000.1, 1.0)
    # Graphene D, G, 2D bands (Lorentzian)
    peaks = [(1350, 200, 30), (1580, 800, 25), (2700, 400, 40)]
    y = np.full_like(x, 50.0)
    for c, a, w in peaks:
        y += a * (w**2 / ((x - c) ** 2 + w**2))
    np.random.seed(42)
    y += np.random.normal(0, 5, size=len(x))

    with FIXTURE.open("w") as f:
        f.write("# Synthetic graphene Raman: D + G + 2D\n")
        f.write("# raman_shift_cm-1   intensity\n")
        for xi, yi in zip(x, y):
            f.write(f"{xi:.1f}   {yi:.2f}\n")


@pytest.fixture(autouse=True)
def _fixture_setup() -> None:
    _generate_raman_fixture()


def test_parse_raman_basic() -> None:
    result = parse_raman(FIXTURE.read_text())
    assert result["spectrum_type"] == "raman"
    assert len(result["peaks"]) >= 3, "Should detect D, G, 2D"


def test_parse_raman_carbon_analysis() -> None:
    result = parse_raman(FIXTURE.read_text())
    carbon = result["carbon_analysis"]
    assert carbon is not None, "Should detect D + G bands"
    assert 0.05 < carbon["id_ig_ratio"] < 1.0
