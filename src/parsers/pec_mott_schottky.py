"""PEC Mott-Schottky (capacitance-voltage) parser for semiconductor photoelectrodes.

New (R220). Reads a Mott-Schottky measurement and computes the semiconductor
figures of merit for a photoelectrode (e.g. WO3, BiVO4, Fe2O3):
  - carrier type (n/p) from the sign of the 1/C^2 vs E slope
  - donor/acceptor density N_D (or N_A) from the slope magnitude (needs eps_r)
  - flat-band potential E_fb from the x-intercept (vs reference and vs RHE)
  - space-charge (depletion) layer width at the most anodic fitted potential

Input shapes (single AC frequency; 2 numeric columns used):
  1. E, C          (capacitance) -> 1/C^2 computed internally
  2. E, 1/C^2      (pre-computed) -> used directly
Capacitance is taken as area-normalized (F/cm^2 or uF/cm^2), so the A^2 term is
absorbed and N is reported directly (confirmed convention, R220). Multi-frequency
files are out of scope here (upload one scan per frequency).

Potentials are converted to the RHE scale only when the reference electrode and
pH are given (no silent assumption), mirroring lsv.py / tafel.py.

Mott-Schottky equation (C per unit area):
    1/C^2 = (2 / (e * eps_r * eps0 * N)) * (E - E_fb - kT/e)

Scientific methods: docs/scientific-methods/pec-analysis.md (section 3)
Refs: Gartner Phys.Rev. 116, 84 (1959); Hankin et al. J.Mater.Chem.A 7, 26162
(2019) DOI:10.1039/C9TA09569A; Coridan et al. EES 8, 2886 (2015).

@phase R220 (PEC Mott-Schottky parser) — photoelectrochemistry cluster.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.parsers._utils import downsample_curve, load_xy

logger = logging.getLogger(__name__)

# Physical constants (SI).
E_CHARGE = 1.602_176_634e-19     # C
EPS0 = 8.854_187_8128e-12        # F/m
KB = 1.380_649e-23               # J/K

# Reference-electrode potential vs SHE/RHE offset (V) at 25 C.
# E_RHE = E_measured + offset + 0.059 * pH   (matches lsv.py / tafel.py)
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

NERNST_SLOPE = 0.059             # V per pH unit at 25 C

# Linear-region search gates.
_R2_FLOOR = 0.997
_MIN_PTS = 6


def _to_inverse_c2(
    e: np.ndarray, c_or_inv: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """Return 1/C^2 (cm^4 F^-2) from the second column, with conversion notes.

    Heuristic on magnitude (single area-normalized column):
      median > 1e6   -> already 1/C^2 (cm^4/F^2), pass through
      median 1e-3..1 -> capacitance in uF/cm^2 -> F/cm^2 -> 1/C^2
      otherwise      -> capacitance in F/cm^2 -> 1/C^2
    """
    notes: list[str] = []
    med = float(np.nanmedian(np.abs(c_or_inv)))
    if med > 1e6:
        notes.append("Second column read as 1/C^2 (cm^4 F^-2).")
        return c_or_inv.astype(float), notes
    if 1e-3 < med < 1e3:
        notes.append("Second column read as capacitance (uF/cm^2); converted to 1/C^2.")
        c_f = c_or_inv * 1e-6
        return 1.0 / c_f**2, notes
    notes.append("Second column read as capacitance (F/cm^2); converted to 1/C^2.")
    return 1.0 / c_or_inv**2, notes


def _fit_line(e: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    m, b = np.polyfit(e, y, 1)
    yhat = m * e + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(m), float(b), r2


def _auto_linear_region(
    e: np.ndarray, y: np.ndarray
) -> tuple[int, int, float, float, float]:
    """Suggest the Mott-Schottky linear region.

    Score = (length / n) * R^2**4. The R^2**4 term penalizes curvature so the
    near-onset bend is excluded, while the length term favors a real plateau
    over a lucky short window. Suggestion only; the app Range Selector lets the
    scientist drag to override (client OLS on this same curve, per ADR-041).
    """
    n = len(e)
    if n < _MIN_PTS:
        m, b, r2 = _fit_line(e, y)
        return 0, n - 1, m, b, r2
    best = (0, _MIN_PTS - 1, *_fit_line(e[:_MIN_PTS], y[:_MIN_PTS]))
    best_score = -1.0
    for i in range(0, n - _MIN_PTS + 1):
        for j in range(i + _MIN_PTS - 1, n):
            m, b, r2 = _fit_line(e[i : j + 1], y[i : j + 1])
            if r2 < _R2_FLOOR:
                continue
            score = (j - i + 1) / n * r2**4
            if score > best_score:
                best_score = score
                best = (i, j, m, b, r2)
    return best


def _carrier_density_cm3(slope_cm4_f2_v: float, eps_r: float) -> float:
    """N from MS slope. slope in cm^4 F^-2 V^-1 (C per area).

    1 cm^4 = 1e-8 m^4, so slope_SI = slope * 1e-8.
    N[m^-3] = 2 / (e * eps_r * eps0 * |slope_SI|); * 1e-6 -> cm^-3.
    """
    slope_si = abs(slope_cm4_f2_v) * 1e-8
    n_m3 = 2.0 / (E_CHARGE * eps_r * EPS0 * slope_si)
    return n_m3 * 1e-6


def _depletion_width_nm(
    e_fb_ref: float, e_anodic: float, n_cm3: float, eps_r: float
) -> float | None:
    n_m3 = n_cm3 * 1e6
    dv = abs(e_anodic - e_fb_ref)
    if n_m3 <= 0 or dv <= 0:
        return None
    w_m = float(np.sqrt(2 * eps_r * EPS0 * dv / (E_CHARGE * n_m3)))
    return round(w_m * 1e9, 3)


def parse_pec_mott_schottky(
    raw_text: str,
    *,
    eps_r: float | None = None,
    reference: str | None = None,
    ph: float | None = None,
    area_cm2: float | None = None,
    temperature_k: float = 298.15,
    frequencies_hz: list[float] | None = None,
) -> dict[str, Any]:
    """Analyse a single-frequency Mott-Schottky scan.

    eps_r is REQUIRED for carrier density — there is no universal value for WO3
    (anisotropic, phase-dependent); without it N is not computed and is flagged.
    """
    notes: list[str] = []

    e, c_col = load_xy(raw_text, min_rows=_MIN_PTS)
    inv_c2, conv_notes = _to_inverse_c2(e, c_col)
    notes.extend(conv_notes)

    # Sort by potential so the linear-region scan and intercept are well defined.
    order = np.argsort(e)
    e, inv_c2 = e[order], inv_c2[order]

    i0, i1, slope, intercept, r2 = _auto_linear_region(e, inv_c2)
    carrier_type = "n-type" if slope > 0 else "p-type"

    kt_over_e = KB * temperature_k / E_CHARGE          # volts
    x_intercept = -intercept / slope                   # where 1/C^2 = 0
    e_fb_ref = x_intercept - kt_over_e

    analysis: dict[str, Any] = {
        "carrier_type": carrier_type,
        "flat_band_V_vs_ref": float(round(e_fb_ref, 4)),
        "x_intercept_V": float(round(x_intercept, 4)),
        "slope_cm4_F2_V": float(slope),
        "fit_r2": float(round(r2, 4)),
        "fit_range_V": [float(round(e[i0], 4)), float(round(e[i1], 4))],
        "fit_range_idx": [int(i0), int(i1)],
        "fit_n_points": int(i1 - i0 + 1),
    }

    # Carrier density (needs eps_r).
    density: float | None = None
    if eps_r is not None and eps_r > 0:
        density = _carrier_density_cm3(slope, eps_r)
        key = "donor_density_cm3" if carrier_type == "n-type" else "acceptor_density_cm3"
        analysis[key] = float(f"{density:.4e}")
        analysis["depletion_width_nm"] = _depletion_width_nm(
            e_fb_ref, float(e[i1]), density, eps_r
        )
    else:
        notes.append(
            "Dielectric constant (eps_r) not provided: carrier density not "
            "computed. No universal value for WO3 (anisotropic, phase-dependent) "
            "— provide dielectricConstant to obtain N_D."
        )

    # Flat-band on the RHE scale (needs reference + pH), same convention as LSV.
    e_fb_rhe: float | None = None
    if reference is not None and ph is not None:
        offset = REFERENCE_OFFSET_V.get(reference.strip().lower().replace(" ", "_"))
        if offset is not None:
            e_fb_rhe = e_fb_ref + offset + NERNST_SLOPE * ph
            analysis["flat_band_V_vs_rhe"] = float(round(e_fb_rhe, 4))
        else:
            notes.append(f"Unknown reference electrode '{reference}'; E_fb vs RHE not computed.")
    else:
        notes.append(
            "Flat-band reported vs reference only: provide referenceElectrode "
            "and pH to convert to the RHE scale."
        )

    if (i1 - i0 + 1) < _MIN_PTS + 1:
        notes.append(
            f"Linear region is short ({i1 - i0 + 1} points); verify the range "
            "with the Range Selector."
        )
    if frequencies_hz and len(frequencies_hz) > 1:
        notes.append(
            "Multiple frequencies supplied but this parser fits a single scan; "
            "upload one measurement per frequency to check flat-band convergence."
        )
    else:
        notes.append(
            "Single-frequency measurement: flat-band may shift with AC frequency; "
            "multi-frequency confirmation recommended (Hankin JMCA 2019)."
        )

    return {
        "spectrum_type": "pec_mott_schottky",
        "peaks": [],
        "spectrum_curve": downsample_curve(e, inv_c2, target_points=500),
        "mott_schottky_curve": {
            "x": [float(round(v, 4)) for v in e.tolist()],
            "y": [float(round(v, 4)) for v in inv_c2.tolist()],
        },
        "analysis": analysis,
        "conditions": {
            "eps_r": eps_r,
            "reference": reference,
            "pH": ph,
            "area_cm2": area_cm2,
            "temperature_k": temperature_k,
        },
        "notes": notes,
        "quick_stats": {
            "rowCount": len(e),
            "xRange": [float(round(e.min(), 4)), float(round(e.max(), 4))],
            "yRange": [float(round(inv_c2.min(), 4)), float(round(inv_c2.max(), 4))],
            "peakCount": 0,
        },
        "x_unit": "V",
        "y_unit": "1/C^2",
    }
