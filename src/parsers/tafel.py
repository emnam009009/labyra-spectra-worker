"""Tafel analysis for HER/OER electrocatalysis.

Reads a polarization curve (potential, current) and derives the kinetic
parameters that a Tafel plot provides but a single LSV figure-of-merit does not:

  - Tafel slope b (mV/dec) from the linear region of eta vs log10|j|
  - exchange current density j0 (extrapolation of the Tafel line to eta = 0)
  - transfer coefficient alpha (from b = 2.303 RT / (alpha F))
  - rate-determining-step hint from the slope value (HER mechanism)

This complements lsv.py: LSV reports the overpotential at 10 mA/cm2 (the primary
activity benchmark) and a Tafel slope; tafel.py focuses on the *mechanism and
kinetics* (j0, alpha, rate-determining step), which McCrory deliberately left
out of the benchmark because it is system-specific (JACS 2013, ref 69).

Scientific basis: Bard, Faulkner & White, Electrochemical Methods, 3rd ed.,
Section 15.2.2 (Tafel plot analysis of HER kinetics). At 25 C the Tafel slope is
b = 2.303 RT / (alpha F) = 0.0592 / alpha V/dec, so alpha ~ 0.5 gives ~118-120
mV/dec, the signature of a rate-limiting Volmer step.

@phase R260 (Tafel parser) — electrochemistry cluster.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.parsers._utils import downsample_curve, load_xy

logger = logging.getLogger(__name__)

# Reference-electrode potential vs SHE/RHE offset (V) at 25 C (mirror lsv.py).
# E_RHE = E_measured + offset + 0.059 * pH
REFERENCE_OFFSET_V: dict[str, float] = {
    "ag/agcl": 0.197,
    "ag/agcl_sat_kcl": 0.197,
    "ag/agcl_3m_kcl": 0.210,
    "sce": 0.241,
    "hg/hgo": 0.140,
    "hg/hgo_1m": 0.140,
    "rhe": 0.0,
    "she": 0.0,
    "nhe": 0.0,
}

NERNST_SLOPE = 0.059          # V per pH unit at 25 C
E_OER_EQ_RHE = 1.23
E_HER_EQ_RHE = 0.0
RT_F_2303 = 0.0592            # 2.303 RT / F at 25 C (V)

# HER rate-determining-step diagnostics by Tafel slope (mV/dec), Bard 15.2.2.
# Approximate textbook values; real systems vary, so reported as a hint.
_HER_MECHANISM = [
    (40.0, "~30 mV/dec: rate-limiting Tafel recombination step (2 Hads -> H2)"),
    (80.0, "~40 mV/dec: rate-limiting Heyrovsky step (Hads + H+ + e- -> H2)"),
    (200.0, "~120 mV/dec: rate-limiting Volmer step (H+ + e- -> Hads); alpha~0.5"),
]


def _parse_two_column(text: str) -> tuple[np.ndarray, np.ndarray]:
    return load_xy(
        text,
        validate=lambda e, i: abs(e).max() < 10,
        min_rows=10,
    )


def _to_rhe(
    e: np.ndarray, reference: str | None, ph: float | None, notes: list[str]
) -> np.ndarray | None:
    if reference is None or ph is None:
        notes.append(
            "Potential not converted to RHE: reference electrode and/or pH "
            "unknown; overpotential and j0 not computed."
        )
        return None
    offset = REFERENCE_OFFSET_V.get(reference.strip().lower())
    if offset is None:
        notes.append(f"Unknown reference electrode '{reference}'; eta not computed.")
        return None
    return e + offset + NERNST_SLOPE * ph


def _current_density(
    i: np.ndarray, area_cm2: float | None
) -> tuple[np.ndarray, str]:
    if area_cm2 and area_cm2 > 0:
        return i / area_cm2 * 1000.0, "mA/cm2"  # A -> mA/cm2
    return i, "raw"


def _mechanism_hint(slope_mv: float, reaction: str) -> str:
    if reaction == "oer":
        return (
            f"{slope_mv:.0f} mV/dec (OER). OER is a 4-electron multistep process; "
            "slopes are commonly ~40-120 mV/dec and do not map cleanly to a single "
            "rate-determining step. Interpret mechanism only with a full kinetic "
            "study (Bard 15.2; McCrory JACS 2013 ref 69)."
        )
    for threshold, label in _HER_MECHANISM:
        if slope_mv < threshold:
            return label
    return (
        f"{slope_mv:.0f} mV/dec is high; may indicate mass-transport limitation, "
        "uncompensated resistance, or a non-ideal/multistep process."
    )


def _tafel_fit(
    eta: np.ndarray, j: np.ndarray, reaction: str
) -> dict[str, Any] | None:
    """
    Fit eta = a + b*log10|j| over the most linear window (R^2 max) in the
    kinetic region. Returns slope b (V/dec), intercept a, R^2, j0, alpha.
    """
    absj = np.abs(j)
    sign = np.sign(np.nanmean(eta)) or 1.0
    mask = (absj > 1e-3) & (eta * sign > 0)  # kinetic branch, |j| > 1 uA
    if mask.sum() < 5:
        return None
    x = np.log10(absj[mask])
    y = eta[mask]
    order = np.argsort(x)
    x, y = x[order], y[order]

    # slide a window of >=5 points, keep the fit with the highest R^2
    best = None
    n = len(x)
    win = max(5, n // 3)
    for start in range(0, n - win + 1):
        xs = x[start:start + win]
        ys = y[start:start + win]
        b, a = np.polyfit(xs, ys, 1)
        yhat = a + b * xs
        ss_res = float(np.sum((ys - yhat) ** 2))
        ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        if best is None or r2 > best["r2"]:
            best = {"b": float(b), "a": float(a), "r2": r2,
                    "lo": float(xs[0]), "hi": float(xs[-1])}
    if best is None or abs(best["b"]) < 1e-9:
        return None

    b = best["b"]
    a = best["a"]
    slope_mv = abs(b) * 1000.0
    # j0: extrapolate Tafel line to eta = 0 -> log10|j0| = -a / b
    log_j0 = -a / b
    j0 = float(10.0 ** log_j0)
    # transfer coefficient from |b| = 0.0592 / alpha at 25 C
    alpha = float(RT_F_2303 / abs(b))

    out: dict[str, Any] = {
        "tafel_slope_mV_per_dec": round(slope_mv, 1),
        "exchange_current_density_j0": j0,
        "j0_unit": "mA/cm2",
        "transfer_coefficient_alpha": round(alpha, 3),
        "r_squared": round(best["r2"], 4),
        "log_j_window": [round(best["lo"], 2), round(best["hi"], 2)],
        "mechanism_hint": _mechanism_hint(slope_mv, reaction),
    }
    return out


def parse_tafel(
    raw_text: str,
    *,
    reference: str | None = None,
    ph: float | None = None,
    area_cm2: float | None = None,
    reaction: str | None = None,   # "her" | "oer"
) -> dict[str, Any]:
    """Analyse a polarization curve for Tafel kinetics (j0, alpha, mechanism).

    Provide reference electrode + pH for RHE conversion and a correct
    overpotential axis; provide area_cm2 so j0 is a current *density*. Without
    these, a Tafel slope can still be reported but j0/alpha are withheld or
    approximate (flagged in notes).
    """
    notes: list[str] = []
    e_raw, i = _parse_two_column(raw_text)
    j, j_unit = _current_density(i, area_cm2)

    e_rhe = _to_rhe(e_raw, reference, ph, notes)
    analysis: dict[str, Any] = {}

    if e_rhe is not None and reaction in ("her", "oer"):
        if reaction == "oer":
            eta = e_rhe - E_OER_EQ_RHE
        else:  # her
            eta = E_HER_EQ_RHE - e_rhe
        if j_unit != "mA/cm2":
            notes.append(
                "Electrode area unknown: j0 is in raw current units, not a "
                "density; treat j0 as approximate."
            )
        fit = _tafel_fit(eta, j, reaction)
        if fit is None:
            notes.append(
                "No clear linear Tafel region found; data may be too noisy, too "
                "short, or dominated by mass transport."
            )
        else:
            analysis["tafel"] = fit
            if fit["r_squared"] < 0.98:
                notes.append(
                    f"Tafel fit R^2 = {fit['r_squared']:.3f} (<0.98): the linear "
                    "window is imperfect; verify the kinetic region and iR "
                    "correction before quoting j0/alpha."
                )
            notes.append(
                "Tafel parameters are mechanism indicators, not the activity "
                "benchmark; report overpotential at 10 mA/cm2 (LSV) as the "
                "primary figure of merit (McCrory JACS 2013)."
            )
    else:
        notes.append(
            "Reaction and/or RHE scale unknown: Tafel slope may be computed from "
            "raw potential but j0/alpha/mechanism are not reliable. Provide "
            "reaction (her/oer), reference electrode, and pH."
        )

    return {
        "spectrum_curve": downsample_curve(e_raw, j, target_points=500),
        "tafel_curve": _build_tafel_curve(e_rhe, j, reaction),
        "analysis": analysis,
        "quick_stats": {
            "rowCount": len(e_raw),
            "eRange_V": [float(round(e_raw.min(), 3)), float(round(e_raw.max(), 3))],
            "current_unit": j_unit,
        },
        "notes": notes,
    }


def _build_tafel_curve(
    e_rhe: np.ndarray | None, j: np.ndarray, reaction: str | None
) -> dict[str, list[float]] | None:
    """The proper Tafel-plot axes: x = log10|j|, y = overpotential (V), restricted
    to the kinetic branch (eta in the driving direction, |j| > 1 uA). Returned so
    the app can render a real Tafel plot AND run an instant client-side linear fit
    over a user-selected window without re-deriving the RHE/density chain.
    """
    if e_rhe is None or reaction not in ("her", "oer"):
        return None
    eta = (e_rhe - E_OER_EQ_RHE) if reaction == "oer" else (E_HER_EQ_RHE - e_rhe)
    absj = np.abs(j)
    mask = (absj > 1e-3) & (eta > 0)
    if mask.sum() < 5:
        return None
    logj = np.log10(absj[mask])
    y = eta[mask]
    order = np.argsort(logj)
    logj, y = logj[order], y[order]
    return {
        "x": [float(round(v, 4)) for v in logj.tolist()],
        "y": [float(round(v, 4)) for v in y.tolist()],
    }
