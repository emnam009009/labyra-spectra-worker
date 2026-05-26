"""Shared utility: downsample spectrum to ~N points for visualization."""

from __future__ import annotations

import re
from io import StringIO
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


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


_NUMERIC_START = re.compile(r"^\s*[+-]?(\d|\.\d)")


def strip_header(text: str) -> str:
    """Keep only numeric data rows.

    Vendor exports (CorrWare/CView, ZPlot/ZView, Gamry, Bio-Logic, PerkinElmer,
    Bruker, Horiba) prepend long text headers that are not '#'-commented; a data
    row starts with a number (optionally signed, optionally a leading dot). Lines
    failing that test are dropped. If nothing matches (already clean), the input
    is returned unchanged so pre-cleaned data is never harmed.

    @phase R256 (universal loader)
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    data = [ln for ln in lines if _NUMERIC_START.match(ln)]
    return "\n".join(data) if data else text


def load_xy(
    text: str,
    *,
    validate: Callable[[np.ndarray, np.ndarray], bool] | None = None,
    min_rows: int = 10,
    min_cols: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Universal two-column loader for spectra/voltammetry text exports.

    Strips vendor headers, normalises EU decimals, then tries common delimiters
    (comma, semicolon, whitespace, tab). Returns the first two numeric columns
    (x, y) that pass the per-technique ``validate(x, y)`` callback. The callback
    encodes the physically valid range (e.g. XRD 0.1-180 deg, FTIR 300-5000
    cm-1) so a wrong column layout is rejected rather than silently accepted.

    Raises ValueError if no delimiter yields a valid table.

    @phase R256 (universal loader) — single source of truth replacing the
    per-parser _parse_two_column copies.
    """
    import pandas as pd  # local import keeps _utils light for edge runtime

    cleaned = normalize_decimal(strip_header(text))
    for sep in [",", ";", r"\s+", "\t"]:
        try:
            df = pd.read_csv(
                StringIO(cleaned), sep=sep, header=None, comment="#",
                engine="python", skip_blank_lines=True,
            )
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            if df.shape[1] < min_cols or len(df) < min_rows:
                continue
            x = df.iloc[:, 0].to_numpy(dtype=float)
            y = df.iloc[:, 1].to_numpy(dtype=float)
            if validate is None or validate(x, y):
                return x, y
        except Exception:
            continue
    raise ValueError("Could not parse two-column data (need numeric x, y columns)")
