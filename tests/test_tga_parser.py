"""Smoke test TGA parser."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.parsers.tga import parse_tga

FIXTURE = Path(__file__).parent / "fixtures" / "tga_sample.csv"


def _generate_tga_fixture() -> None:
    if FIXTURE.exists() and FIXTURE.stat().st_size > 200:
        return
    T = np.arange(25.0, 800.0, 0.5)
    # Synthetic 3-stage decomposition:
    # 25-150: water loss (5%), 150-400: organic (25%), 400-600: oxide (10%)
    mass = np.full_like(T, 100.0)
    # Stage 1: water (sigmoid centered at 100)
    mass -= 5.0 / (1 + np.exp(-(T - 100) / 15))
    # Stage 2: organic (sigmoid at 280)
    mass -= 25.0 / (1 + np.exp(-(T - 280) / 30))
    # Stage 3: oxide (sigmoid at 500)
    mass -= 10.0 / (1 + np.exp(-(T - 500) / 25))
    np.random.seed(42)
    mass += np.random.normal(0, 0.05, size=len(T))

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE.open("w") as f:
        f.write("# Synthetic TGA 3-stage\n# temp_C,mass_%\n")
        for t, m in zip(T, mass):
            f.write(f"{t:.2f},{m:.3f}\n")


@pytest.fixture(autouse=True)
def _setup() -> None:
    _generate_tga_fixture()


def test_parse_tga_basic() -> None:
    result = parse_tga(FIXTURE.read_text())
    assert result["spectrum_type"] == "tga"
    assert result["temp_unit"] == "C"
    assert "dtg_curve" in result


def test_parse_tga_stages() -> None:
    result = parse_tga(FIXTURE.read_text())
    stages = result["decomp_stages"]
    # Should find 2-3 stages (water, organic, oxide)
    assert len(stages) >= 2
    # Total loss ~40%
    assert 30 < result["total_loss_pct"] < 50
