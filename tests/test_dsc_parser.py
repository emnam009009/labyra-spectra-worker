"""Smoke test DSC parser."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.parsers.dsc import parse_dsc

FIXTURE = Path(__file__).parent / "fixtures" / "dsc_sample.csv"


def _generate_dsc_fixture() -> None:
    if FIXTURE.exists() and FIXTURE.stat().st_size > 200:
        return
    T = np.arange(25.0, 350.0, 0.5)
    hf = np.zeros_like(T)
    # Endothermic peak (Tm) at 180 deg-C
    hf -= 5.0 * np.exp(-((T - 180) ** 2) / (2 * 15**2))
    # Exothermic peak (Tc) at 250 deg-C
    hf += 3.0 * np.exp(-((T - 250) ** 2) / (2 * 12**2))
    np.random.seed(42)
    hf += np.random.normal(0, 0.05, size=len(T))

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE.open("w") as f:
        f.write("# Synthetic DSC, Tm=180, Tc=250\n# temp_C,heat_flow_mW\n")
        for t, h in zip(T, hf):
            f.write(f"{t:.2f},{h:.4f}\n")


@pytest.fixture(autouse=True)
def _setup() -> None:
    _generate_dsc_fixture()


def test_parse_dsc_basic() -> None:
    result = parse_dsc(FIXTURE.read_text())
    assert result["spectrum_type"] == "dsc"


def test_parse_dsc_peaks() -> None:
    result = parse_dsc(FIXTURE.read_text())
    assert len(result["endothermic_peaks"]) >= 1, "Should detect Tm"
    assert len(result["exothermic_peaks"]) >= 1, "Should detect Tc"
    # Tm around 180
    tm_peaks = [p["peak_T"] for p in result["endothermic_peaks"]]
    assert any(170 < t < 190 for t in tm_peaks)
