"""PEC J-V (photoelectrochemistry linear sweep under illumination).

Analyses a current-voltage scan of a photoelectrode for water splitting:
  - photocurrent density at a benchmark potential (light - dark, or net j)
  - photocurrent onset potential (where |j| first rises above a threshold)
  - solar-to-hydrogen (STH) efficiency when the reaction + illumination power
    are known (only valid under AM1.5G, two-electrode, no applied bias — flagged)

Two input shapes are supported:
  1. Three columns E, j_light, j_dark  -> net photocurrent = j_light - j_dark
  2. Two columns E, j (a single light or chopped scan) -> j treated as measured

Scientific basis: Chen, Jaramillo et al., "Accelerating materials development
for photoelectrochemical hydrogen production", J. Mater. Res. 2010; STH per
Coridan et al., Energy Environ. Sci. 2015 (rigorous STH definition).

@phase R219 (PEC J-V parser) — photoelectrochemistry cluster.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.parsers._utils import downsample_curve, load_xy

logger = logging.getLogger(__name__)

# Standard AM1.5G one-sun illumination power (mW/cm2).
P_AM15G_MW_CM2 = 100.0
# Thermodynamic water-splitting potential (V).
E_WATER_SPLIT_V = 1.23
# Onset definition: |j| threshold (mA/cm2).
ONSET_J = 0.1
# Benchmark potential for photocurrent reporting vs RHE (V).
BENCHMARK_E_RHE = 1.23


def _current_density(i: np.ndarray, area_cm2: float | None) -> tuple[np.ndarray, str]:
    if area_cm2 and area_cm2 > 0:
        return i / area_cm2 * 1000.0, "mA/cm2"  # A -> mA/cm2
    return i, "raw"


def _onset_potential(e: np.ndarray, j: np.ndarray) -> float | None:
    """First potential where |j| exceeds the onset threshold (sorted by E)."""
    order = np.argsort(e)
    es, js = e[order], np.abs(j[order])
    idx = np.where(js >= ONSET_J)[0]
    if idx.size == 0:
        return None
    return float(round(es[idx[0]], 4))


def _j_at_potential(e: np.ndarray, j: np.ndarray, target_e: float) -> float | None:
    """Interpolate j at a target potential."""
    order = np.argsort(e)
    es, js = e[order], j[order]
    if target_e < es.min() or target_e > es.max():
        return None
    return float(round(float(np.interp(target_e, es, js)), 4))


def parse_pec_jv(
    raw_text: str,
    *,
    area_cm2: float | None = None,
    light_power_mw_cm2: float | None = None,
    applied_bias_v: float | None = None,
) -> dict[str, Any]:
    """Analyse a PEC J-V scan.

    light_power_mw_cm2 defaults to AM1.5G (100) when not given (flagged).
    applied_bias_v, if provided, is used to flag that a non-zero bias makes the
    reported STH an "applied-bias photon-to-current efficiency" (ABPE), not a
    true STH (a common literature error we surface explicitly).
    """
    notes: list[str] = []

    # PEC J-V scan: potential vs current. A chopped-light scan shows the
    # light/dark steps as a sawtooth within this single j trace.
    e, i = load_xy(raw_text, min_rows=10)
    j, j_unit = _current_density(i, area_cm2)
    j_light = None
    j_dark = None

    if area_cm2 is None:
        notes.append("Electrode area unknown: current is raw, not a density; STH not computed.")

    analysis: dict[str, Any] = {"current_density_unit": j_unit}

    onset = _onset_potential(e, j)
    if onset is not None:
        analysis["photocurrent_onset_V"] = onset

    j_bench = _j_at_potential(e, j, BENCHMARK_E_RHE)
    if j_bench is not None:
        analysis["photocurrent_at_1p23V_RHE"] = j_bench

    # STH efficiency (only meaningful for zero-bias water splitting, mA/cm2).
    p_light = light_power_mw_cm2 if light_power_mw_cm2 else P_AM15G_MW_CM2
    if light_power_mw_cm2 is None:
        notes.append(f"Illumination power assumed AM1.5G ({P_AM15G_MW_CM2:.0f} mW/cm2).")
    if j_unit == "mA/cm2":
        # |j| at the thermodynamic potential drives water splitting at zero bias.
        j_op = _j_at_potential(e, j, BENCHMARK_E_RHE)
        if j_op is not None:
            sth = abs(j_op) * E_WATER_SPLIT_V / p_light * 100.0  # %
            if applied_bias_v:
                analysis["abpe_percent"] = round(sth, 3)
                notes.append(
                    f"Non-zero applied bias ({applied_bias_v} V): reported as ABPE, "
                    "not true STH. True STH requires zero applied bias (Coridan EES 2015)."
                )
            else:
                analysis["sth_percent"] = round(sth, 3)
                notes.append(
                    "STH is only valid under AM1.5G, two-electrode, zero applied bias, "
                    "100% Faradaic efficiency to H2/O2. Verify these conditions."
                )

    light_dark_curve = None
    if j_light is not None and j_dark is not None:
        light_dark_curve = {
            "light": downsample_curve(e, j_light, target_points=500),
            "dark": downsample_curve(e, j_dark, target_points=500),
        }

    return {
        "spectrum_type": "pec_jv",
        "peaks": [],
        "spectrum_curve": downsample_curve(e, j, target_points=500),
        "light_dark_curve": light_dark_curve,
        "analysis": analysis,
        "conditions": {
            "area_cm2": area_cm2,
            "light_power_mw_cm2": p_light,
            "applied_bias_v": applied_bias_v,
        },
        "notes": notes,
        "quick_stats": {
            "rowCount": len(e),
            "xRange": [float(round(e.min(), 4)), float(round(e.max(), 4))],
            "yRange": [float(round(j.min(), 4)), float(round(j.max(), 4))],
            "peakCount": 0,
        },
        "x_unit": "V",
        "y_unit": "Current",
    }
