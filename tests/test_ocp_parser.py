"""Smoke test OCP parser."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.parsers.ocp import parse_ocp

FIXTURE = Path(__file__).parent / "fixtures" / "ocp_sample.csv"


def _generate_ocp_fixture() -> None:
    if FIXTURE.exists() and FIXTURE.stat().st_size > 200:
        return
    t = np.arange(0.0, 600.0, 1.0)
    # Initial rise, then plateau at +0.25 V
    v = 0.25 - 0.15 * np.exp(-t / 100.0)
    np.random.seed(42)
    v += np.random.normal(0, 0.001, size=len(t))

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE.open("w") as f:
        f.write("# Synthetic OCP, eq=0.25V, 10 min\n# time_s,potential_V\n")
        for ti, vi in zip(t, v):
            f.write(f"{ti:.1f},{vi:.5f}\n")


@pytest.fixture(autouse=True)
def _setup() -> None:
    _generate_ocp_fixture()


def test_parse_ocp_basic() -> None:
    result = parse_ocp(FIXTURE.read_text())
    assert result["spectrum_type"] == "ocp"


def test_parse_ocp_equilibrium() -> None:
    result = parse_ocp(FIXTURE.read_text())
    eq = result["equilibrium"]
    assert 0.20 < eq["equilibrium_potential_V"] < 0.30
    assert eq["stability"] in ("stable", "drifting")
