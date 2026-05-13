"""Simulate XRD powder pattern from CIF using Dans_Diffraction.

Returns discrete peaks (2θ, intensity) with symmetry-equivalent peaks merged.

Notes:
  - Cu-Kα1 (1.5406 Å = 8.04778 keV) default
  - 2θ range 10-80° (typical XRD scan)
  - hkl indices generated up to ±max_h cube, intensities computed,
    grouped by 2θ (round to 0.1°) and intensities summed (multiplicity).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ENERGY_KEV = 8.04778  # Cu-Kα1
DEFAULT_MIN_TWOTHETA = 10.0
DEFAULT_MAX_TWOTHETA = 80.0
HKL_MAX = 6  # generate hkl in cube [-6, 6] = 2197 reflections
PEAK_GROUP_TOLERANCE_DEG = 0.1


def simulate_powder_pattern(
    cif_text: str,
    *,
    energy_kev: float = DEFAULT_ENERGY_KEV,
    min_twotheta: float = DEFAULT_MIN_TWOTHETA,
    max_twotheta: float = DEFAULT_MAX_TWOTHETA,
    relative_intensity_threshold: float = 0.02,
) -> list[dict[str, Any]]:
    """Compute simulated powder pattern from CIF text.

    Returns list of peak dicts: {twotheta, intensity, relative_intensity, multiplicity, hkl}
    sorted by 2θ ascending.
    """
    # Import locally to avoid loading Dans_Diffraction unless citation is used
    try:
        import Dans_Diffraction as dif  # type: ignore[import-untyped]
    except ImportError as exc:
        logger.error("Dans_Diffraction not available: %s", exc)
        return []

    with tempfile.NamedTemporaryFile(mode="w", suffix=".cif", delete=False) as f:
        f.write(cif_text)
        cif_path = f.name

    try:
        xtl = dif.Crystal(cif_path)
        xtl.Scatter.setup_scatter(
            scattering_type="xray",
            energy_kev=energy_kev,
            min_twotheta=min_twotheta,
            max_twotheta=max_twotheta,
            output=False,
        )

        # Generate hkl grid
        hkl_list = []
        for h in range(-HKL_MAX, HKL_MAX + 1):
            for k in range(-HKL_MAX, HKL_MAX + 1):
                for l in range(-HKL_MAX, HKL_MAX + 1):
                    if (h, k, l) == (0, 0, 0):
                        continue
                    hkl_list.append([h, k, l])
        hkl_arr = np.array(hkl_list)

        # Compute 2θ (may produce NaN for q > 4π/λ → arcsin out of range)
        with np.errstate(invalid="ignore"):
            tth = xtl.Cell.tth(hkl_arr, energy_kev)
        mask = np.isfinite(tth) & (tth >= min_twotheta) & (tth <= max_twotheta)
        hkl_arr = hkl_arr[mask]
        tth = tth[mask]
        if len(hkl_arr) == 0:
            return []

        # Compute intensity per reflection
        I = xtl.Scatter.intensity(hkl_arr)
        if I is None or len(I) != len(hkl_arr):
            return []

        I_max = float(np.max(I)) if len(I) > 0 else 0.0
        if I_max <= 0:
            return []

        # Group by 2θ (round to 0.1°) → multiplicity-weighted sum
        bins: dict[float, dict[str, Any]] = {}
        for t, i_val, h_vec in zip(tth, I, hkl_arr):
            if i_val < I_max * 0.001:  # filter extremely weak
                continue
            key = round(float(t) / PEAK_GROUP_TOLERANCE_DEG) * PEAK_GROUP_TOLERANCE_DEG
            if key not in bins:
                bins[key] = {
                    "twotheta": float(t),
                    "intensity": 0.0,
                    "multiplicity": 0,
                    "hkl": h_vec.tolist(),
                }
            bins[key]["intensity"] += float(i_val)
            bins[key]["multiplicity"] += 1

        # Convert to list, normalize relative intensity
        peaks = sorted(bins.values(), key=lambda p: p["twotheta"])
        I_max_grouped = max(p["intensity"] for p in peaks) if peaks else 1.0
        filtered_peaks: list[dict[str, Any]] = []
        for p in peaks:
            rel = p["intensity"] / I_max_grouped
            if rel < relative_intensity_threshold:
                continue
            filtered_peaks.append({
                "twotheta": round(p["twotheta"], 3),
                "intensity": round(p["intensity"], 1),
                "relative_intensity": round(rel, 4),
                "multiplicity": p["multiplicity"],
                "hkl": p["hkl"],
            })
        return filtered_peaks
    except Exception as exc:  # noqa: BLE001
        logger.warning("XRD simulation failed: %s", exc)
        return []
    finally:
        Path(cif_path).unlink(missing_ok=True)
