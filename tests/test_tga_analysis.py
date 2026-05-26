"""
TGA thermal-analysis regression tests (R248).

Covers ISO 11358-1 extrapolated onset (tangent ∩ baseline, distinct from the
DTG peak), stability indices T5%/T10%/T50%, char yield, and multi-stage
decomposition. Synthetic sigmoidal mass-loss curves.
"""

from __future__ import annotations

import numpy as np

from src.parsers.tga import parse_tga


def _to_text(x: np.ndarray, y: np.ndarray) -> str:
    return "\n".join(f"{xi:.1f},{yi:.3f}" for xi, yi in zip(x, y, strict=False))


def _single_stage() -> str:
    """100% -> 40% sigmoidal loss centred at 375 °C; char 40%."""
    t = np.arange(30.0, 700.0, 1.0)
    mass = 40 + 60 / (1 + np.exp((t - 375) / 15))
    return _to_text(t, mass)


def _two_stage() -> str:
    """15% loss ~200 °C, then 55% loss ~450 °C; char 30%."""
    t = np.arange(30.0, 800.0, 1.0)
    mass = 30 + 15 / (1 + np.exp((t - 200) / 12)) + 55 / (1 + np.exp((t - 450) / 18))
    return _to_text(t, mass)


# --- extrapolated onset (the ISO 11358 guard) -------------------------------

def test_extrapolated_onset_below_peak() -> None:
    """ISO extrapolated onset must sit below the DTG peak temperature."""
    r = parse_tga(_single_stage())
    s = r["decomp_stages"][0]
    assert s["extrapolated_onset_T"] is not None
    assert s["extrapolated_onset_T"] < s["peak_T"]


def test_extrapolated_above_deviation_onset() -> None:
    """Extrapolated onset is higher than the first-deflection (deviation) onset."""
    r = parse_tga(_single_stage())
    s = r["decomp_stages"][0]
    assert s["extrapolated_onset_T"] > s["deviation_onset_T"]


def test_peak_is_max_rate_temperature() -> None:
    """DTG peak ≈ sigmoid centre (375 °C) = temperature of maximum rate."""
    r = parse_tga(_single_stage())
    assert abs(r["decomp_stages"][0]["peak_T"] - 375.0) < 10.0


# --- stability indices ------------------------------------------------------

def test_stability_indices_ordered() -> None:
    """T5% < T10% < T50% (monotone cumulative loss)."""
    s = parse_tga(_single_stage())["stability"]
    assert s["T5_pct"] < s["T10_pct"] < s["T50_pct"]


def test_char_yield_matches_final_mass() -> None:
    r = parse_tga(_single_stage())
    assert abs(r["char_yield_pct"] - 40.0) < 1.0
    assert r["char_yield_pct"] == r["final_mass_pct"]


# --- multi-stage ------------------------------------------------------------

def test_two_stage_detected() -> None:
    r = parse_tga(_two_stage())
    stages = r["decomp_stages"]
    assert len(stages) == 2
    # stage losses roughly 15% then 55%
    losses = sorted(s["mass_loss_pct"] for s in stages)
    assert 10 < losses[0] < 20
    assert 48 < losses[1] < 60


def test_two_stage_onsets_increasing() -> None:
    r = parse_tga(_two_stage())
    onsets = [s["extrapolated_onset_T"] for s in r["decomp_stages"]]
    assert onsets[0] < onsets[1]
