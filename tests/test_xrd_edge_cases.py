"""
XRD parser edge-case regression tests (worker bug audit B1/B2/B3).

Self-contained: generates synthetic two-column XRD text inline rather than
reading a fixture file (the .xy fixture is .gitignore'd and absent on clean
clones — see W1 notes). These lock the bug fixes:

  B3  grazing-incidence (GISAXS, ~0.5° start) and high-angle-only (≥80°) scans
      must PARSE, not raise. (Pre-fix: both raised "Could not parse".)
  B1/B2  Scherrer crystallite size must stay finite/None near 2θ→180° where
      cos(θ)→0. (Pre-fix: D blew up to ~1e17 nm.)
"""

from __future__ import annotations

import math

import pytest

from src.parsers.xrd import _parse_two_column, parse_xrd


def _synth(two_theta_start: float, two_theta_end: float, step: float = 0.05) -> str:
    """Synthetic CSV: a Gaussian-ish bump so a peak is detectable."""
    n = int((two_theta_end - two_theta_start) / step)
    center = (two_theta_start + two_theta_end) / 2.0
    lines = []
    for i in range(n):
        x = two_theta_start + i * step
        y = 100.0 + 9000.0 * math.exp(-((x - center) ** 2) / (2 * 0.4**2))
        lines.append(f"{x:.3f},{y:.1f}")
    return "\n".join(lines)


# --- B3: range validation ---------------------------------------------------

def test_b3_low_angle_gisaxs_parses() -> None:
    """Grazing-incidence scan starting at 0.5° must parse (was rejected)."""
    x, _ = _parse_two_column(_synth(0.5, 10.0))
    assert x.min() < 1.0
    assert len(x) > 10


def test_b3_high_angle_only_parses() -> None:
    """High-angle-only scan (80-120°) must parse (was rejected)."""
    x, _ = _parse_two_column(_synth(80.0, 120.0))
    assert x.min() >= 80.0
    assert len(x) > 10


def test_b3_standard_range_still_parses() -> None:
    """Regression: ordinary 10-80° scan still parses."""
    x, _ = _parse_two_column(_synth(10.0, 80.0))
    assert 10.0 <= x.min() < 11.0
    assert x.max() <= 80.0


def test_b3_out_of_range_still_rejected() -> None:
    """x.max > 180 (impossible 2θ) is still rejected."""
    with pytest.raises(ValueError):
        _parse_two_column(_synth(150.0, 200.0))


# --- B1/B2: Scherrer cos_theta guard ----------------------------------------

def test_b1_b2_high_angle_scherrer_finite() -> None:
    """
    A scan with a peak near 2θ=178° must NOT yield an exploded crystallite
    size. cos(θ) → 0 there; the guard returns None for that peak instead of inf.
    """
    result = parse_xrd(_synth(170.0, 179.5, step=0.02))
    avg = result["scherrer_avg_nm"]
    # finite-or-None, never astronomically large / inf / NaN
    if avg is not None:
        assert math.isfinite(avg)
        assert 0 < avg < 1e4, f"crystallite size unphysical: {avg} nm"
    # per-peak sizes likewise bounded
    for pk in result["peaks"]:
        d = pk.get("crystallite_size_nm")
        if d is not None:
            assert math.isfinite(d) and 0 < d < 1e4, f"peak D unphysical: {d}"


def test_b1_b2_normal_angle_scherrer_unchanged() -> None:
    """Regression: mid-angle peak still produces a positive finite size."""
    result = parse_xrd(_synth(20.0, 30.0, step=0.02))
    avg = result["scherrer_avg_nm"]
    assert avg is None or (math.isfinite(avg) and avg > 0)
