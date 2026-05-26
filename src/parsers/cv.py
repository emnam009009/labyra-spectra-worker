"""CV (Cyclic Voltammetry) parser.

New (R254). Splits the forward/reverse sweep, locates anodic/cathodic peaks,
and computes the redox descriptors:
  - Epa, Epc, peak currents ipa, ipc
  - peak separation dEp = Epa - Epc (reversibility: ~59/n mV is Nernstian)
  - formal potential E0' = (Epa + Epc)/2
  - peak current ratio |ipa/ipc| (~1 for reversible)
  - reversibility classification (one scan rate is provisional; multi-scan-rate
    needed to confirm)

ECSA / Randles-Sevcik need a scan-rate series (not a single CV); flagged, not
fabricated.

Scientific methods: docs/scientific-methods/cv-analysis.md
"""

from __future__ import annotations

import logging
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src.parsers._utils import downsample_curve, normalize_decimal

logger = logging.getLogger(__name__)

NERNST_DEP_MV = 59.0  # 59/n mV ideal peak separation at 25 C


def _parse_two_column(text: str) -> tuple[np.ndarray, np.ndarray]:
    text = normalize_decimal(text)
    for sep in [",", ";", r"\s+", "\t"]:
        try:
            df = pd.read_csv(
                StringIO(text), sep=sep, header=None, comment="#",
                engine="python", skip_blank_lines=True,
            )
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            if df.shape[1] >= 2 and len(df) > 20:
                e = df.iloc[:, 0].to_numpy(dtype=float)
                i = df.iloc[:, 1].to_numpy(dtype=float)
                if abs(e).max() < 10:
                    return e, i
        except Exception:
            continue
    raise ValueError("Could not parse two-column CV data (potential, current)")


def _split_sweeps(e: np.ndarray, i: np.ndarray) -> tuple[
    tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
]:
    """
    Split into anodic (E increasing, up to the positive vertex) and cathodic
    (E decreasing, after the vertex) sweeps using the most-positive potential.
    """
    apex = int(np.argmax(e))
    return (e[: apex + 1], i[: apex + 1]), (e[apex:], i[apex:])


def _peak_in_branch(
    e: np.ndarray, i: np.ndarray, anodic: bool
) -> tuple[float, float] | None:
    """Dominant peak in a branch. Anodic = max +current; cathodic = max -current."""
    if len(e) < 5:
        return None
    sig = i if anodic else -i
    base = float(np.median(sig))
    prom = (sig.max() - sig.min()) * 0.05
    idx, _ = find_peaks(sig, prominence=max(prom, 1e-12))
    if len(idx) == 0:
        k = int(np.argmax(sig))
        if sig[k] - base <= 0:
            return None
        return float(e[k]), float(i[k])
    k = int(idx[np.argmax(sig[idx])])
    return float(e[k]), float(i[k])


def _reversibility(dep_mv: float | None, ratio: float | None) -> str:
    if dep_mv is None:
        return "indeterminate (single peak)"
    if dep_mv <= 80 and (ratio is None or 0.8 <= ratio <= 1.25):
        return "reversible-like (dEp near 59/n mV, ipa/ipc ~ 1)"
    if dep_mv <= 200:
        return "quasi-reversible (confirm with scan-rate dependence)"
    return "irreversible-like (large dEp)"


def parse_cv(
    raw_text: str,
    n_electrons: int = 1,
    scan_rate_v_s: float | None = None,
    area_cm2: float | None = None,
) -> dict[str, Any]:
    e, i = _parse_two_column(raw_text)
    (e_fwd, i_fwd), (e_rev, i_rev) = _split_sweeps(e, i)

    anodic = _peak_in_branch(e_fwd, i_fwd, anodic=True)
    cathodic = _peak_in_branch(e_rev, i_rev, anodic=False)

    notes: list[str] = []
    analysis: dict[str, Any] = {}

    epa = ipa = epc = ipc = None
    if anodic:
        epa, ipa = anodic
        analysis["Epa_V"] = float(round(epa, 4))
        analysis["ipa"] = float(round(ipa, 8))
    if cathodic:
        epc, ipc = cathodic
        analysis["Epc_V"] = float(round(epc, 4))
        analysis["ipc"] = float(round(ipc, 8))

    dep_mv = None
    ratio = None
    if epa is not None and epc is not None:
        dep_mv = float(round((epa - epc) * 1000.0, 1))
        analysis["dEp_mV"] = dep_mv
        analysis["E0_prime_V"] = float(round((epa + epc) / 2.0, 4))
        analysis["dEp_ideal_mV"] = round(NERNST_DEP_MV / n_electrons, 1)
        if ipc not in (None, 0):
            ratio = float(round(abs(ipa / ipc), 3))
            analysis["peak_current_ratio"] = ratio
    else:
        notes.append("Only one peak resolved; redox couple incomplete (dEp/E0' not computed).")

    analysis["reversibility"] = _reversibility(dep_mv, ratio)
    notes.append(
        "Reversibility from a single scan rate is provisional; confirm with "
        "scan-rate dependence (dEp vs v, ip vs sqrt(v))."
    )

    if scan_rate_v_s is None:
        notes.append("Scan rate unknown: Randles-Sevcik (ip vs sqrt(v)) and ECSA need a scan-rate series.")
    if area_cm2 is None:
        notes.append("Electrode area unknown: current densities and ECSA not computed.")

    return {
        "spectrum_type": "cv",
        "peaks": [],
        "spectrum_curve": downsample_curve(e, i, target_points=600),
        "analysis": analysis,
        "conditions": {
            "n_electrons": n_electrons,
            "scan_rate_v_s": scan_rate_v_s,
            "area_cm2": area_cm2,
        },
        "notes": notes,
        "quick_stats": {
            "rowCount": len(e),
            "xRange": [float(round(e.min(), 4)), float(round(e.max(), 4))],
            "yRange": [float(round(i.min(), 8)), float(round(i.max(), 8))],
            "peakCount": int(bool(anodic)) + int(bool(cathodic)),
        },
        "x_unit": "V",
        "y_unit": "Current",
    }
