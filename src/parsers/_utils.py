"""Shared utility: downsample spectrum to ~N points for visualization."""

from __future__ import annotations

import re

import numpy as np


def downsample_curve(
    x: np.ndarray,
    y: np.ndarray,
    *,
    target_points: int = 500,
) -> dict[str, list[float]]:
    """Reduce a spectrum to ~target_points uniformly spaced.

    Returns {"x": [...], "y": [...]} ready for JSON serialization.
    Preserves min/max of y in each bucket to keep peak visibility.
    """
    n = len(x)
    if n <= target_points:
        return {
            "x": [float(round(v, 4)) for v in x.tolist()],
            "y": [float(round(v, 6)) for v in y.tolist()],
        }

    # Bin-and-pick: in each bin, take min and max y values
    bucket_size = max(1, n // target_points)
    out_x: list[float] = []
    out_y: list[float] = []
    for i in range(0, n, bucket_size):
        chunk_x = x[i : i + bucket_size]
        chunk_y = y[i : i + bucket_size]
        if len(chunk_x) == 0:
            continue
        # Take min then max within bucket to preserve peaks
        min_idx = int(np.argmin(chunk_y))
        max_idx = int(np.argmax(chunk_y))
        pairs = sorted([(min_idx, chunk_y[min_idx]), (max_idx, chunk_y[max_idx])])
        for local_idx, val in pairs:
            out_x.append(float(round(chunk_x[local_idx], 4)))
            out_y.append(float(round(val, 6)))

    return {"x": out_x, "y": out_y}


def normalize_decimal(text: str) -> str:
    """Convert EU decimal comma (e.g. "1,523") to dot, so EU-locale instrument
    exports (PerkinElmer/Bruker/Horiba) parse instead of coercing to NaN.

    Conservative: only rewrites when a NON-comma delimiter (tab or semicolon) is
    present, so comma cannot be the column separator. This avoids corrupting
    comma-delimited integer CSV like "400,1523". Pure ASCII heuristic, no deps.

    @phase R246-W2 (audit B4)
    """
    sample = [ln for ln in text.splitlines()[:30] if ln.strip()]
    if not sample:
        return text
    uses_tab = any("\t" in ln for ln in sample)
    uses_semicolon = any(";" in ln for ln in sample)
    has_comma_decimal = any(re.search(r"\d,\d", ln) for ln in sample)
    has_dot_decimal = any(re.search(r"\d\.\d", ln) for ln in sample)
    if has_comma_decimal and not has_dot_decimal and (uses_tab or uses_semicolon):
        return re.sub(r"(\d),(\d)", r"\1.\2", text)
    return text
