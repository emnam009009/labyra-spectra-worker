"""Shared utility: downsample spectrum to ~N points for visualization."""

from __future__ import annotations

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
