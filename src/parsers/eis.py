"""EIS (Electrochemical Impedance Spectroscopy) parser.

New (R250). Two-tier analysis:
  1. Model-free readout (always runs, robust): Rs, Rct, Cdl, Warburg flag, and
     exchange current density j0 — read directly from the Nyquist data.
  2. Equivalent-circuit fit (optional, impedance.py): Randles R0-p(R1,CPE1)
     [-W1], seeded with the model-free estimates so it rarely diverges.

Input: 2-4 columns. Accepts (freq, Z', Z'') or (freq, |Z|, phase_deg). The
sign convention of Z'' varies by instrument; this parser normalises so the
capacitive arc has Z'' < 0 (Nyquist plots -Z'' upward).

Scientific methods: docs/scientific-methods/eis-analysis.md
"""

from __future__ import annotations

import logging
import re
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd

from src.parsers._utils import downsample_curve, normalize_decimal

logger = logging.getLogger(__name__)

# Faraday / gas constants for exchange current density.
R_GAS = 8.314462618      # J/(mol·K)
F_FARADAY = 96485.332    # C/mol


def _autodetect_columns(arr: np.ndarray) -> tuple[int, int, int]:
    """
    Pick (freq_col, zre_col, zim_col) from a multi-column table (e.g. ZPlot/Gamry
    export with Freq, Ampl, Bias, Time, Z', Z'', ...). Heuristic grounded in the
    physics: Z'' is predominantly negative (capacitive), Z' is the adjacent
    column before it, freq is the wide-range positive column (default col 0).
    """
    ncol = arr.shape[1]
    # Z'' = column with the largest fraction of negative values (>30%).
    neg_frac = [float((arr[:, c] < 0).mean()) for c in range(ncol)]
    zim_col = int(np.argmax(neg_frac))
    if neg_frac[zim_col] < 0.3:
        # no clearly-capacitive column → fall back to (0, 1, 2)
        return 0, 1, 2
    # Z' = the adjacent column just before Z'' (ZPlot/Gamry/Autolab convention),
    # provided it is non-constant; otherwise search backward for one.
    zre_col = zim_col - 1
    while zre_col > 0 and float(np.std(arr[:, zre_col])) < 1e-9:
        zre_col -= 1
    # freq: wide log-range positive column, prefer col 0.
    freq_col = 0
    return freq_col, zre_col, zim_col


_NUMERIC_START = re.compile(r"^\s*[+-]?(\d|\.\d)")


def _strip_header(text: str) -> str:
    """
    Keep only numeric data rows. Vendor exports (ZPlot/ZView, Gamry, Autolab)
    prepend long text headers that are not '#'-commented; a data row starts with
    a number (optionally signed). Lines failing that test are dropped.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    data = [ln for ln in lines if _NUMERIC_START.match(ln)]
    return "\n".join(data) if data else text


def _parse_columns(
    text: str, data_format: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse EIS into (freq_Hz, Z_real, Z_imag).

    Column layouts:
      data_format='polar':            freq, |Z|, phase(deg) -> Z', Z''
      otherwise (default): rectangular. For exactly 3 columns -> freq, Z', Z''.
      For >3 columns (ZPlot/Gamry/Autolab exports with Ampl/Bias/Time/GD/Range
      mixed in) -> columns auto-detected (Z'' = most-negative column, Z' before
      it, freq = col 0).

    Header / comment lines (non-numeric) are skipped automatically. Z'' is
    normalised so the capacitive arc is negative (Nyquist plots -Z'' up).
    """
    text = normalize_decimal(_strip_header(text))
    for sep in [",", ";", r"\s+", "\t"]:
        try:
            df = pd.read_csv(
                StringIO(text), sep=sep, header=None, comment="#",
                engine="python", skip_blank_lines=True,
            )
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            if df.shape[1] < 3 or len(df) < 8:
                continue
            arr = df.to_numpy(dtype=float)
            if data_format == "polar":
                f, z_mod, phase = arr[:, 0], arr[:, 1], np.radians(arr[:, 2])
                z_real = z_mod * np.cos(phase)
                z_imag = z_mod * np.sin(phase)
            elif df.shape[1] == 3:
                f, z_real, z_imag = arr[:, 0], arr[:, 1], arr[:, 2]
            else:
                fc, zrc, zic = _autodetect_columns(arr)
                f, z_real, z_imag = arr[:, fc], arr[:, zrc], arr[:, zic]
            if (f <= 0).any():
                continue
            # Normalise sign so the capacitive arc is negative.
            if np.nanmean(z_imag) > 0:
                z_imag = -z_imag
            return f, z_real, z_imag
        except Exception:
            continue
    raise ValueError("Could not parse EIS data (need >=3 columns: freq, Z', Z'')")


def _model_free_readout(
    f: np.ndarray, zr: np.ndarray, zi: np.ndarray, area_cm2: float | None, n_electrons: int,
    temperature_k: float,
) -> dict[str, Any]:
    """Read Rs, Rct, Cdl, Warburg, j0 directly from the Nyquist data."""
    # order by frequency descending (high f first) for stable end-picks
    order = np.argsort(-f)
    fo, zro, zio = f[order], zr[order], zi[order]
    neg_zi = -zio  # positive in the capacitive arc

    # Rs: real-axis intercept at the highest frequency.
    rs = float(zro[0])

    # Semicircle apex = max(-Z''). Cdl from omega at apex: omega_max = 1/(Rct·Cdl).
    apex = int(np.argmax(neg_zi))
    f_apex = float(fo[apex])

    # Rct = diameter of the semicircle. Take the real-axis value where the arc
    # closes after the apex (local minimum of -Z'' past the apex), which excludes
    # any low-frequency Warburg tail. Fall back to the last point.
    rct_real = float(zro[-1])
    if apex < len(neg_zi) - 2:
        tail = neg_zi[apex:]
        # first local-ish minimum after the apex (arc returning toward the axis)
        close_rel = int(np.argmin(tail))
        rct_real = float(zro[apex + close_rel])
    rct = float(round(max(rct_real - rs, 1e-9), 4))

    cdl = None
    arc_incomplete = bool(apex >= len(fo) - 2)  # apex at lowest-freq end => arc not closed
    if f_apex > 0 and rct > 0 and not arc_incomplete:
        omega_max = 2 * np.pi * f_apex
        cdl = float(1.0 / (omega_max * rct))

    # Warburg: 45° tail at low frequency → slope of Z'' vs Z' near unity.
    warburg = False
    if len(fo) >= 6:
        lo = slice(-max(5, len(fo) // 5), None)
        dzr = np.diff(zro[lo])
        dzi = np.diff(neg_zi[lo])
        with np.errstate(divide="ignore", invalid="ignore"):
            slopes = dzi / dzr
        med_slope = float(np.nanmedian(slopes))
        warburg = 0.6 < med_slope < 1.6  # ~1 (45°)

    # Exchange current density j0 = R·T / (n·F·Rct·A) (needs area).
    j0 = None
    if area_cm2 and rct > 0:
        # Rct·A gives area-specific resistance (Ω·cm²); j0 in A/cm².
        rct_area = rct * area_cm2
        j0 = float(R_GAS * temperature_k / (n_electrons * F_FARADAY * rct_area))

    return {
        "Rs_ohm": float(round(rs, 4)),
        "Rct_ohm": rct,
        "Cdl_F": float(round(cdl, 12)) if cdl else None,
        "f_apex_Hz": float(round(f_apex, 4)),
        "arc_incomplete": arc_incomplete,
        "warburg_detected": warburg,
        "exchange_current_density_A_cm2": float(round(j0, 9)) if j0 else None,
    }


def _circuit_fit(
    f: np.ndarray, zr: np.ndarray, zi: np.ndarray, seed: dict[str, Any], warburg: bool,
) -> dict[str, Any] | None:
    """Randles fit via impedance.py, seeded from the model-free readout."""
    try:
        from impedance.models.circuits import CustomCircuit
    except ImportError:
        return {"error": "impedance.py not installed"}

    rs = max(seed["Rs_ohm"], 1e-3)
    rct = max(seed["Rct_ohm"], 1e-3)
    cdl = seed["Cdl_F"] or 1e-5

    if warburg:
        circuit_str = "R0-p(R1,CPE1)-W1"
        guess = [rs, rct, cdl, 0.9, max(rct * 0.5, 1.0)]
    else:
        circuit_str = "R0-p(R1,CPE1)"
        guess = [rs, rct, cdl, 0.9]

    z = zr + 1j * zi
    try:
        circuit = CustomCircuit(circuit_str, initial_guess=guess)
        circuit.fit(f, z)
        names, _units = circuit.get_param_names()
        params = {nm: float(round(v, 9)) for nm, v in zip(names, circuit.parameters_, strict=False)}
        z_pred = circuit.predict(f, use_initial=False)
        # normalised chi-square (sum of squared residuals / |Z|²)
        resid = np.abs(z - z_pred) ** 2 / np.maximum(np.abs(z) ** 2, 1e-12)
        chi_sq = float(round(float(np.sum(resid)), 6))
        return {"circuit": circuit_str, "parameters": params, "chi_square": chi_sq}
    except Exception as exc:  # fit can diverge; report instead of crash
        return {"circuit": circuit_str, "error": f"fit failed: {type(exc).__name__}"}


def parse_eis(
    raw_text: str,
    area_cm2: float | None = None,
    n_electrons: int = 1,
    temperature_k: float = 298.15,
    data_format: str | None = None,
    do_fit: bool = True,
) -> dict[str, Any]:
    f, zr, zi = _parse_columns(raw_text, data_format=data_format)
    readout = _model_free_readout(f, zr, zi, area_cm2, n_electrons, temperature_k)

    notes: list[str] = []
    if area_cm2 is None:
        notes.append("Electrode area unknown: exchange current density j0 not computed.")
    if readout.get("arc_incomplete"):
        notes.append(
            "Semicircle not closed within the measured frequency range: Rct is a "
            "lower bound and Cdl is not estimated. Extend to lower frequency for a "
            "reliable fit."
        )

    fit_result = None
    if do_fit:
        fit_result = _circuit_fit(f, zr, zi, readout, readout["warburg_detected"])
        if fit_result and "error" in fit_result:
            notes.append(f"Equivalent-circuit fit unavailable: {fit_result['error']}")
        elif fit_result and fit_result.get("chi_square", 0) > 1.0:
            notes.append(
                f"Circuit fit poor (chi-square={fit_result['chi_square']:.2g}); "
                "parameters unreliable — check data quality / circuit model."
            )

    # Nyquist curve (Z' vs -Z'') for display
    nyquist = {"z_real": [float(round(v, 4)) for v in zr.tolist()],
               "z_imag_neg": [float(round(-v, 4)) for v in zi.tolist()]}

    return {
        "spectrum_type": "eis",
        "peaks": [],
        "nyquist": nyquist,
        "bode_curve": downsample_curve(f, np.abs(zr + 1j * zi), target_points=500),
        "model_free": readout,
        "circuit_fit": fit_result,
        "conditions": {
            "area_cm2": area_cm2,
            "n_electrons": n_electrons,
            "temperature_K": temperature_k,
        },
        "notes": notes,
        "quick_stats": {
            "rowCount": len(f),
            "xRange": [float(round(f.min(), 4)), float(round(f.max(), 4))],
            "yRange": [float(round(zr.min(), 2)), float(round(zr.max(), 2))],
            "peakCount": 0,
        },
        "x_unit": "Hz",
        "y_unit": "Ohm",
    }
