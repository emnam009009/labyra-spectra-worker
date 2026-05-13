"""Smoke test UV-Vis DRS parser."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.parsers.uvvis_drs import parse_uvvis_drs

FIXTURE = Path(__file__).parent / "fixtures" / "uvvis_drs_sample.csv"


def _generate_drs_fixture() -> None:
    if FIXTURE.exists() and FIXTURE.stat().st_size > 200:
        return
    wavelength = np.arange(200.0, 800.1, 0.5)
    energy = 1240.0 / wavelength
    edge_ev = 2.7
    # Reflectance: low for E > Eg (absorbing), high for E < Eg (transparent)
    # R drops from 0.9 (transparent) to 0.1 (strongly absorbing) across edge
    R = 0.9 - 0.8 * np.where(energy > edge_ev, 1.0, 0.0)
    # Add some structure
    R = R + 0.05 * np.exp(-((wavelength - 350) / 30) ** 2)  # small bump
    R = np.clip(R, 0.05, 0.95)
    np.random.seed(42)
    R += np.random.normal(0, 0.005, size=len(wavelength))
    R = np.clip(R, 0.01, 0.99)
    # Save as %R (0-100)
    R_pct = R * 100

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE.open("w") as f:
        f.write("# DRS synthetic, bandgap=2.7eV\n# wavelength_nm,reflectance_%\n")
        for w, r in zip(wavelength, R_pct):
            f.write(f"{w:.2f},{r:.3f}\n")


@pytest.fixture(autouse=True)
def _setup() -> None:
    _generate_drs_fixture()


def test_parse_drs_basic() -> None:
    result = parse_uvvis_drs(FIXTURE.read_text())
    assert result["spectrum_type"] == "uvvis_drs"
    assert result["reflectance_mode"] == "percent"
    assert "reflectance_curve" in result
    assert "km_curve" in result
    assert "tauc_curve" in result


def test_parse_drs_bandgap() -> None:
    result = parse_uvvis_drs(FIXTURE.read_text())
    tauc = result["tauc_bandgap"]
    assert tauc is not None
    assert 2.0 < tauc["bandgap_ev"] < 3.5
