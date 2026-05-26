"""LSV (Linear Sweep Voltammetry) parser for HER/OER electrocatalysis.

New (R253). Reads a polarization curve (potential, current) and computes the
benchmarking figures of merit for water-splitting electrocatalysts:
  - overpotential at 10 mA/cm2 (geometric) — the standard activity benchmark
  - onset potential (at 1 mA/cm2)
  - Tafel slope (mV/dec) from the linear region of eta vs log|j|

Potentials are converted to the RHE scale when the reference electrode and pH
are given (eta cannot be computed otherwise — no silent assumption).

Scientific methods: docs/scientific-methods/lsv-analysis.md
"""

from __future__ import annotations

import logging
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd

from src.parsers._utils import downsample_curve, normalize_decimal

logger = logging.getLogger(__name__)

# Reference-electrode potential vs SHE/RHE offset (V) at 25 C.
# E_RHE = E_measured + offset + 0.059 * pH
REFERENCE_OFFSET_V: dict[str, float] = {
    "ag/agcl": 0.197,            # saturated KCl (common default)
    "ag/agcl_sat_kcl": 0.197,
    "ag/agcl_3m_kcl": 0.210,
    "sce": 0.241,                # saturated calomel
    "hg/hgo": 0.140,             # 1 M (alkaline)
    "hg/hgo_1m": 0.140,
    "rhe": 0.0,
    "she": 0.0,
    "nhe": 0.0,
}

NERNST_SLOPE = 0.059             # V per pH unit at 25 C
E_OER_EQ_RHE = 1.23             # O2/H2O equilibrium vs RHE
E_HER_EQ_RHE = 0.0              # H+/H2 equilibrium vs RHE

BENCHMARK_J = 10.0              # mA/cm2 — standard activity benchmark
ONSET_J = 1.0                  # mA/cm2 — onset definition


def _parse_two_column(text: str) -> tuple[np.ndarray, np.ndarray]:
    text = normalize_decimal(text)
    for sep in [",", ";", r"\s+", "\t"]:
        try:
            df = pd.read_csv(
                StringIO(text), sep=sep, header=None, comment="#",
                engine="python", skip_blank_lines=True,
            )
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            if df.shape[1] >= 2 and len(df) > 10:
                e = df.iloc[:, 0].to_numpy(dtype=float)
                i = df.iloc[:, 1].to_numpy(dtype=float)
                # potential typically -2..+2 V; current any scale
                if abs(e).max() < 10:
                    return e, i
        except Exception:
            continue
    raise ValueError("Could not parse two-column LSV data (potential, current)")


def _to_rhe(
    e: np.ndarray, reference: str | None, ph: float | None
) -> tuple[np.ndarray | None, list[str]]:
    """Convert measured potential to the RHE scale. Needs reference + pH."""
    notes: list[str] = []
    if reference is None or ph is None:
        notes.append(
            "Potential not converted to RHE: reference electrode and/or pH unknown. "
            "Overpotential cannot be computed; provide referenceElectrode and pH."
        )
        return None, notes
    key = reference.strip().lower().replace(" ", "_")
    offset = REFERENCE_OFFSET_V.get(key)
    if offset is None:
        notes.append(f"Unknown reference electrode '{reference}'; eta not computed.")
        return None, notes
    e_rhe = e + offset + NERNST_SLOPE * ph
    return e_rhe, notes


def _current_density(i: np.ndarray, area_cm2: float | None) -> tuple[np.ndarray, str]:
    """mA/cm2 if area given (assume input current in mA); else raw current."""
    if area_cm2 and area_cm2 > 0:
        return i / area_cm2, "mA/cm2"
    return i, "raw"


def _potential_at_j(
    e_rhe: np.ndarray, j: np.ndarray, j_target: float
) -> float | None:
    """First potential where |j| reaches j_target (interpolated)."""
    absj = np.abs(j)
    idx = np.where(absj >= j_target)[0]
    if len(idx) == 0:
        return None
    k = int(idx[0])
    if k == 0:
        return float(round(e_rhe[0], 4))
    j0, j1 = absj[k - 1], absj[k]
    if j1 == j0:
        return float(round(e_rhe[k], 4))
    frac = (j_target - j0) / (j1 - j0)
    return float(round(e_rhe[k - 1] + frac * (e_rhe[k] - e_rhe[k - 1]), 4))


def _tafel_slope(
    eta: np.ndarray, j: np.ndarray
) -> dict[str, Any] | None:
    """
    Tafel slope (mV/dec) from the linear region of eta vs log10|j|.
    Selects the most linear window (R^2 max) over the kinetically meaningful
    range (|j| 0.1-10 mA/cm2 typically). eta = a + b*log10|j|, b in V/dec.
    """
    absj = np.abs(j)
    mask = (absj > 1e-3) & (eta * np.sign(np.nanmean(eta)) > 0)
    x = np.log10(absj[mask])
    y = eta[mask]
    if len(x) < 5:
        return None
    # sort by x and scan windows for the best linear fit
    order = np.argsort(x)
    x, y = x[order], y[order]
    best = None
    n = len(x)
    win = max(5, n // 3)
    for start in range(0, n - win + 1):
        xs, ys = x[start:start + win], y[start:start + win]
        if xs[-1] - xs[0] < 0.3:  # need at least ~0.3 decade span
            continue
        b, a = np.polyfit(xs, ys, 1)
        yhat = a + b * xs
        ss_res = float(np.sum((ys - yhat) ** 2))
        ss_tot = float(np.sum((ys - ys.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        if best is None or r2 > best["r2"]:
            best = {
                "tafel_slope_mV_per_dec": float(round(abs(b) * 1000, 1)),
                "r2": float(round(r2, 4)),
                "log_j_range": [float(round(xs[0], 2)), float(round(xs[-1], 2))],
            }
    return best


def parse_lsv(
    raw_text: str,
    area_cm2: float | None = None,
    reference: str | None = None,
    ph: float | None = None,
    reaction: str | None = None,   # "her" | "oer"
    ir_corrected: bool = False,
) -> dict[str, Any]:
    e, i = _parse_two_column(raw_text)
    j, j_unit = _current_density(i, area_cm2)
    e_rhe, rhe_notes = _to_rhe(e, reference, ph)
    notes: list[str] = list(rhe_notes)

    analysis: dict[str, Any] = {"current_density_unit": j_unit}
    if area_cm2 is None:
        notes.append("Electrode area unknown: current shown as raw; benchmarks need mA/cm2.")

    # Benchmarks require eta (RHE scale + reaction known) and current density.
    eta = None
    if e_rhe is not None and reaction in ("her", "oer") and j_unit == "mA/cm2":
        e_eq = E_OER_EQ_RHE if reaction == "oer" else E_HER_EQ_RHE
        # eta magnitude positive in the driving direction
        eta = (e_rhe - e_eq) if reaction == "oer" else (e_eq - e_rhe)
        analysis["reaction"] = reaction
        analysis["overpotential_at_10mA_cm2_V"] = _potential_at_j(eta, j, BENCHMARK_J)
        analysis["onset_overpotential_at_1mA_cm2_V"] = _potential_at_j(eta, j, ONSET_J)
        tafel = _tafel_slope(eta, j)
        analysis["tafel"] = tafel
        if not ir_corrected:
            notes.append(
                "Not iR-corrected: overpotential and Tafel slope are overestimated; "
                "apply iR compensation for benchmarking."
            )
    elif reaction not in ("her", "oer"):
        notes.append("Reaction type unknown: provide reaction='her' or 'oer' for overpotential benchmarks.")

    return {
        "spectrum_type": "lsv",
        "peaks": [],
        "spectrum_curve": downsample_curve(e, i, target_points=500),
        "rhe_curve": (
            downsample_curve(e_rhe, j, target_points=500) if e_rhe is not None else None
        ),
        "analysis": analysis,
        "conditions": {
            "area_cm2": area_cm2,
            "reference": reference,
            "pH": ph,
            "reaction": reaction,
            "ir_corrected": ir_corrected,
        },
        "notes": notes,
        "quick_stats": {
            "rowCount": len(e),
            "xRange": [float(round(e.min(), 4)), float(round(e.max(), 4))],
            "yRange": [float(round(i.min(), 6)), float(round(i.max(), 6))],
            "peakCount": 0,
        },
        "x_unit": "V",
        "y_unit": "Current",
    }
