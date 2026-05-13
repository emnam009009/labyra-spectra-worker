"""Smoke test for XRD parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.parsers.xrd import parse_xrd


FIXTURE = Path(__file__).parent / "fixtures" / "xrd_wo3_sample.xy"


def test_parse_xrd_returns_peaks() -> None:
    text = FIXTURE.read_text()
    result = parse_xrd(text)

    assert result["spectrum_type"] == "xrd"
    assert "peaks" in result
    assert len(result["peaks"]) >= 3, f"Expected ≥3 peaks, got {len(result['peaks'])}"

    # Top peak should be around 23.6° (WO3 monoclinic strongest)
    top = max(result["peaks"], key=lambda p: p["intensity"])
    assert 22.0 < top["two_theta"] < 25.0, f"Expected top peak in 22-25°, got {top['two_theta']}"


def test_parse_xrd_quick_stats() -> None:
    result = parse_xrd(FIXTURE.read_text())
    qs = result["quick_stats"]
    assert qs["rowCount"] > 30
    assert qs["xRange"][0] < 15
    assert qs["xRange"][1] > 70
    assert qs["peakCount"] >= 1


def test_parse_xrd_scherrer() -> None:
    result = parse_xrd(FIXTURE.read_text())
    # With sparse data (Δ2θ=2°), FWHM resolution is poor; scherrer might be None or large
    assert result["scherrer_avg_nm"] is None or result["scherrer_avg_nm"] > 0


def test_parse_xrd_malformed_raises() -> None:
    with pytest.raises(ValueError):
        parse_xrd("not a valid xrd file\nrandom garbage")
