"""Match user-measured XRD peaks against simulated patterns.

Scoring:
  - Match ratio: count of user peaks aligned with simulated peaks within tolerance
  - Intensity correlation: Pearson r between matched user/simulated intensities
  - Final score: 0.7 * match_ratio + 0.3 * max(0, intensity_correlation)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

PEAK_TOLERANCE_DEG = 0.3  # ±0.3° for peak match


def match_peaks(
    user_peaks: list[dict[str, Any]],
    simulated_peaks: list[dict[str, Any]],
    *,
    tolerance_deg: float = PEAK_TOLERANCE_DEG,
) -> dict[str, Any]:
    """Score how well simulated peaks match user peaks.

    user_peaks: [{two_theta, intensity, ...}, ...]  (from xrd parser)
    simulated_peaks: [{twotheta, intensity, relative_intensity, ...}, ...] (from simulator)

    Returns {match_ratio, matched_count, total_user_peaks, intensity_correlation, score}.
    """
    if not user_peaks or not simulated_peaks:
        return _empty_match(len(user_peaks))

    user_2t = np.array([p["two_theta"] for p in user_peaks])
    user_I = np.array([p["intensity"] for p in user_peaks])
    sim_2t = np.array([p["twotheta"] for p in simulated_peaks])
    sim_rel_I = np.array([p["relative_intensity"] for p in simulated_peaks])

    matched_pairs: list[tuple[int, int]] = []  # (user_idx, sim_idx)
    used_sim = set()
    for ui, ut in enumerate(user_2t):
        # Find closest unused simulated peak within tolerance
        diffs = np.abs(sim_2t - ut)
        for si in np.argsort(diffs):
            si_int = int(si)
            if si_int in used_sim:
                continue
            if diffs[si_int] > tolerance_deg:
                break
            matched_pairs.append((ui, si_int))
            used_sim.add(si_int)
            break

    matched_count = len(matched_pairs)
    match_ratio = matched_count / max(len(user_peaks), 1)

    intensity_correlation: float | None = None
    if matched_count >= 3:
        user_matched_I = user_I[[p[0] for p in matched_pairs]]
        sim_matched_I = sim_rel_I[[p[1] for p in matched_pairs]]
        # Normalize user intensities to [0, 1]
        u_max = user_matched_I.max() if user_matched_I.max() > 0 else 1.0
        user_norm = user_matched_I / u_max
        if user_norm.std() > 0 and sim_matched_I.std() > 0:
            intensity_correlation = float(np.corrcoef(user_norm, sim_matched_I)[0, 1])

    intensity_term = max(0.0, intensity_correlation) if intensity_correlation is not None else 0.0
    score = 0.7 * match_ratio + 0.3 * intensity_term

    # Build hkl assignment: user_peak_idx → hkl from matched sim peak
    user_hkl_map: dict[int, list[int]] = {}
    for user_idx, sim_idx in matched_pairs:
        sim_peak = simulated_peaks[sim_idx]
        hkl = sim_peak.get("hkl")
        if hkl:
            user_hkl_map[user_idx] = hkl

    return {
        "match_ratio": round(match_ratio, 3),
        "matched_count": matched_count,
        "total_user_peaks": len(user_peaks),
        "intensity_correlation": round(intensity_correlation, 3) if intensity_correlation is not None else None,
        "score": round(score, 3),
        "user_hkl_map": user_hkl_map,
    }


def _empty_match(total: int) -> dict[str, Any]:
    return {
        "match_ratio": 0.0,
        "matched_count": 0,
        "total_user_peaks": total,
        "intensity_correlation": None,
        "score": 0.0,
        "user_hkl_map": {},
    }
