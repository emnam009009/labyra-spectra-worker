"""Smoke test UV-Vis parser."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.parsers.uvvis import parse_uvvis

FIXTURE = Path(__file__).parent / "fixtures" / "uvvis_wo3_sample.csv"


def _generate_uvvis_fixture() -> None:
    """Generate synthetic UV-Vis WO3-like data (bandgap ~2.7 eV)."""
    if FIXTURE.exists() and FIXTURE.stat().st_size > 200:
        return
    wavelength = np.arange(200.0, 800.1, 0.5)
    # Absorption edge at ~460 nm (= 2.7 eV bandgap)
    energy = 1240.0 / wavelength
    edge_ev = 2.7
    # Direct gap: α ∝ sqrt(E - Eg) above edge
    absorbance = np.where(energy > edge_ev, 0.5 * np.sqrt(energy - edge_ev), 0.05)
    # Add small noise
    np.random.seed(42)
    absorbance += np.random.normal(0, 0.005, size=len(wavelength))

    with FIXTURE.open("w") as f:
        f.write("# UV-Vis synthetic WO3-like, bandgap=2.7 eV direct\n")
        f.write("# wavelength_nm,absorbance\n")
        for w, a in zip(wavelength, absorbance):
            f.write(f"{w:.2f},{a:.5f}\n")


@pytest.fixture(autouse=True)
def _fixture_setup() -> None:
    _generate_uvvis_fixture()


def test_parse_uvvis_basic() -> None:
    text = FIXTURE.read_text()
    result = parse_uvvis(text)
    assert result["spectrum_type"] == "uvvis"
    assert result["quick_stats"]["rowCount"] > 1000
    assert result["x_unit"] == "nm"


def test_parse_uvvis_bandgap() -> None:
    result = parse_uvvis(FIXTURE.read_text())
    tauc = result["tauc_bandgap"]
    assert tauc is not None, "Bandgap should be detected"
    # Should be ~2.5-3.0 eV for synthetic WO3
    assert 2.0 < tauc["bandgap_ev"] < 3.5, f"Got {tauc['bandgap_ev']}"
    assert tauc["r_squared"] > 0.8, f"R²={tauc['r_squared']} too low"


def test_parse_uvvis_malformed() -> None:
    with pytest.raises(ValueError):
        parse_uvvis("not csv\nrandom garbage")
