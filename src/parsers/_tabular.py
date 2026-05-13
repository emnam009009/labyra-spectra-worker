"""Smart 2-column tabular parser for XRD/UV-Vis/Raman/FTIR.

Detects 2theta + intensity columns by header name in CSV/TSV/Excel files.
Supports multi-column files with descriptive headers.
"""

from __future__ import annotations

import logging
import re
from io import BytesIO, StringIO
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


X_AXIS_PATTERNS = [
    r"^2[\u03b8t]heta", r"^2[-_\s]?theta", r"^2[\u03b8t]$",
    r"^theta", r"^angle", r"^position",
    r"^wavelength", r"^wavenumber", r"^shift", r"^temperature",
    r"^temp$", r"^t\b", r"^time", r"^energy", r"^x$",
]

Y_AXIS_PATTERNS = [
    r"^intensity", r"^counts$", r"^cps$", r"^i\(", r"^i$",
    r"^raw", r"^absorbance", r"^transmittance", r"^%t",
    r"^reflectance", r"^heat[\s_]?flow", r"^mass", r"^potential",
    r"^y$", r"^y\s",
]


def _matches(name: str, patterns: list[str]) -> bool:
    lower = name.strip().lower()
    return any(re.match(p, lower) for p in patterns)


def _is_skip_col(name: str) -> bool:
    """Index/ID/error columns to skip."""
    lower = name.strip().lower()
    return bool(re.match(r"^(index|id|no\.?|#|number|d\s?\(|sigma|err|\u03c3|cps|time)", lower))


def _pick_xy_columns(df: pd.DataFrame) -> tuple[int, int] | None:
    """Find 2theta + intensity column indices."""
    if df.shape[1] < 2:
        return None

    cols = [str(c) for c in df.columns]

    # All-numeric headers → no real header → first 2 numeric cols
    def is_text(v: Any) -> bool:
        try:
            float(v)
            return False
        except (ValueError, TypeError):
            return True

    if all(not is_text(c) for c in cols):
        return (0, 1)

    # Score columns
    x_idx = next((i for i, c in enumerate(cols) if _matches(c, X_AXIS_PATTERNS)), None)
    y_idx = None
    if x_idx is not None:
        # Find Y after X, skip "skip" columns
        for i in range(len(cols)):
            if i == x_idx:
                continue
            if _matches(cols[i], Y_AXIS_PATTERNS):
                y_idx = i
                break

    # Fallback: skip Index-like cols, use first 2 numeric
    if x_idx is None or y_idx is None:
        usable: list[int] = []
        for i, name in enumerate(cols):
            if _is_skip_col(name):
                continue
            series = pd.to_numeric(df.iloc[:, i], errors="coerce")
            if series.notna().sum() / max(len(series), 1) > 0.8:
                usable.append(i)
            if len(usable) >= 2:
                break
        if len(usable) >= 2:
            if x_idx is None:
                x_idx = usable[0]
            if y_idx is None:
                y_idx = usable[1] if usable[1] != x_idx else (usable[0] if len(usable) > 2 else None)

    if x_idx is None or y_idx is None or x_idx == y_idx:
        return None
    logger.info("Detected columns: x=%d (%s), y=%d (%s)",
                x_idx, cols[x_idx], y_idx, cols[y_idx])
    return (x_idx, y_idx)


def parse_xlsx_two_column(raw_bytes: bytes) -> tuple[np.ndarray, np.ndarray] | None:
    """Parse .xlsx → (x, y) arrays. Returns None on failure."""
    try:
        df = pd.read_excel(BytesIO(raw_bytes), sheet_name=0, engine="openpyxl")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Excel parse failed: %s", exc)
        return None

    pick = _pick_xy_columns(df)
    if not pick:
        return None
    x_idx, y_idx = pick

    x = pd.to_numeric(df.iloc[:, x_idx], errors="coerce")
    y = pd.to_numeric(df.iloc[:, y_idx], errors="coerce")
    mask = x.notna() & y.notna()
    return (x[mask].to_numpy(dtype=float), y[mask].to_numpy(dtype=float))


def parse_csv_two_column(text: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Parse text CSV/TSV with smart column detection. Returns None if can't parse."""
    cleaned_lines = []
    for line in text.split("\n"):
        s = line.strip()
        if not s or s.startswith(("#", "%", ";", "[", "*")):
            continue
        cleaned_lines.append(line)
    if not cleaned_lines:
        return None
    cleaned = "\n".join(cleaned_lines)

    delimiters = [",", "\t", r"\s+", ";"]
    for sep in delimiters:
        try:
            df = pd.read_csv(
                StringIO(cleaned), sep=sep, engine="python",
                skip_blank_lines=True, on_bad_lines="skip",
            )
            if df.shape[1] >= 2 and len(df) >= 10:
                pick = _pick_xy_columns(df)
                if pick:
                    x_idx, y_idx = pick
                    x = pd.to_numeric(df.iloc[:, x_idx], errors="coerce")
                    y = pd.to_numeric(df.iloc[:, y_idx], errors="coerce")
                    mask = x.notna() & y.notna()
                    if mask.sum() >= 10:
                        return (x[mask].to_numpy(dtype=float), y[mask].to_numpy(dtype=float))
        except Exception:  # noqa: BLE001
            continue
    return None
