"""Parse user-pasted XRD reference cards.

Strategy:
1. Extract metadata lines (PDF#, phase name) before peak data
2. Determine column schema from peak data structure:
   - If first peak line has 4+ numeric tokens AND last 3 form valid hkl
     → schema: 2θ d I hkl
   - If 3 numeric tokens AND looks like hkl in col 3 → 2θ I hkl
   - 3 numeric tokens otherwise → 2θ d I
   - 2 numeric tokens → 2θ I
3. Apply same schema to all peak lines
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


PDF_CARD_PATTERN = re.compile(
    r"(PDF-?[24]?\+?\s*#?\s*\d+[-\s]\d+|ICDD\s*\d+[-\s]\d+|JCPDS\s*\d+[-\s]\d+)",
    re.IGNORECASE,
)
FORMULA_PATTERN = re.compile(r"\b((?:[A-Z][a-z]?\d*){2,8})\b")


def _is_hkl_definite(token: str) -> bool:
    """STRICT hkl check: only tokens with explicit hkl markers.

    Definite hkl tokens:
    - Has underscore (was parenthesized): "0_0_2", "-1_2_0"
    - Has minus sign in middle: "1-10", "0-12"

    Ambiguous tokens like "100", "002" handled positionally
    (treated as hkl ONLY when in last column position).
    """
    t = token.strip("()[]")
    if "_" in t:
        parts = t.split("_")
        return len(parts) == 3 and all(re.fullmatch(r"-?\d{1,2}", p) for p in parts)
    # Has explicit minus → hkl
    if "-" in t[1:]:  # minus not at position 0
        return bool(re.fullmatch(r"-?\d-?\d-?\d", t))
    return False


def _looks_like_compact_hkl(token: str) -> bool:
    """Loose hkl check: pure 3-digit string like '002', '100'."""
    return bool(re.fullmatch(r"\d{3}", token))


def _split_columns(line: str) -> list[str]:
    """Split line, protecting parenthesized hkl with underscores."""
    line = re.sub(
        r"[\(\[]\s*(-?\d{1,2})\s+(-?\d{1,2})\s+(-?\d{1,2})\s*[\)\]]",
        r"\1_\2_\3",
        line,
    )
    return [p for p in re.split(r"[\s,;|\t]+", line.strip()) if p]


def _classify_tokens(parts: list[str]) -> tuple[list[float], list[str]]:
    """Split parts into (numeric_tokens, hkl_tokens).

    Strategy:
    - Definite hkl (has _, or -) → always hkl
    - Last token: if "ddd" 3-digit string AND no definite hkl found → hkl
    - Otherwise: try float
    """
    if not parts:
        return [], []

    nums: list[float] = []
    hkls: list[str] = []
    last_token_used_as_hkl = False

    # First pass: catch definite hkls
    for p in parts:
        if _is_hkl_definite(p):
            hkls.append(p.strip("()[]").replace("_", " "))

    # Second pass: handle last token specially if it's ambiguous 3-digit
    last = parts[-1]
    if not hkls and _looks_like_compact_hkl(last) and len(parts) >= 3:
        hkls.append(last)
        last_token_used_as_hkl = True

    # Third pass: collect numbers (skip tokens already used as hkl)
    used_set = set()
    for h in hkls:
        # Original form
        for p in parts:
            if p.strip("()[]").replace("_", " ") == h:
                used_set.add(id(p))
                break
    for i, p in enumerate(parts):
        if last_token_used_as_hkl and i == len(parts) - 1:
            continue
        if _is_hkl_definite(p):
            continue
        try:
            nums.append(float(p))
        except ValueError:
            pass
    return nums, hkls


def _looks_like_peak_line(parts: list[str]) -> bool:
    """Check if line could be a peak: first token is 2θ-looking number."""
    if not parts:
        return False
    try:
        first = float(parts[0])
        return 2 < first < 180
    except ValueError:
        return False


def _detect_schema(peak_lines: list[list[str]]) -> str:
    """Detect column schema from sample of peak lines.

    Returns: "2T_D_I_HKL", "2T_I_HKL", "2T_D_I", or "2T_I".
    """
    # Use first 3 peak lines for detection (majority vote)
    sample = peak_lines[:5]
    has_hkl_count = 0
    has_d_count = 0
    n_tokens_list = []

    for parts in sample:
        nums, hkls = _classify_tokens(parts)
        n_tokens_list.append(len(nums))
        if hkls:
            has_hkl_count += 1
        # d-spacing detection: 2nd number in 0.3-30 range
        if len(nums) >= 3 and 0.3 < nums[1] < 30:
            has_d_count += 1

    has_hkl = has_hkl_count >= len(sample) // 2 + 1
    has_d = has_d_count >= len(sample) // 2 + 1
    typical_n = max(set(n_tokens_list), key=n_tokens_list.count) if n_tokens_list else 2

    if typical_n >= 3 and has_d and has_hkl:
        return "2T_D_I_HKL"
    if typical_n >= 3 and has_d:
        return "2T_D_I"
    if typical_n >= 2 and has_hkl:
        return "2T_I_HKL"
    return "2T_I"


def _parse_peak_with_schema(parts: list[str], schema: str) -> dict[str, Any] | None:
    """Parse peak using detected schema."""
    nums, hkls = _classify_tokens(parts)
    if not nums:
        return None
    two_theta = nums[0]
    if not (2 < two_theta < 180):
        return None

    d_spacing: float | None = None
    intensity: float | None = None
    hkl: str = hkls[0] if hkls else ""

    if schema == "2T_D_I_HKL":
        if len(nums) >= 3:
            d_spacing = nums[1] if 0.3 < nums[1] < 30 else None
            intensity = nums[2]
    elif schema == "2T_D_I":
        if len(nums) >= 3:
            d_spacing = nums[1] if 0.3 < nums[1] < 30 else None
            intensity = nums[2]
        elif len(nums) == 2:
            intensity = nums[1]
    elif schema == "2T_I_HKL":
        if len(nums) >= 2:
            intensity = nums[1]
        # 3rd number (if any) treated as hkl int
        if not hkl and len(nums) >= 3:
            third = int(nums[2])
            if 0 <= third < 1000:
                hkl = str(third).zfill(3)
    else:  # 2T_I
        if len(nums) >= 2:
            intensity = nums[1]

    if intensity is None or not (0 < intensity <= 100):
        return None

    peak = {
        "twoTheta": round(two_theta, 3),
        "intensity": round(intensity, 1),
    }
    if d_spacing is not None:
        peak["dSpacing"] = round(d_spacing, 4)
    if hkl:
        peak["hkl"] = hkl
    return peak


def parse_reference_card(text: str) -> dict[str, Any]:
    """Parse pasted reference card text."""
    if not text or not text.strip():
        raise ValueError("Empty text")

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        raise ValueError("No content")

    # Find peak data lines
    peak_line_idxs: list[int] = []
    parsed_peak_parts: list[list[str]] = []
    for i, line in enumerate(lines):
        parts = _split_columns(line)
        if _looks_like_peak_line(parts):
            peak_line_idxs.append(i)
            parsed_peak_parts.append(parts)

    if not peak_line_idxs:
        raise ValueError("No peak data found (expected 2θ values 2-180°)")

    # Metadata = lines before first peak line
    first_peak_idx = peak_line_idxs[0]
    metadata_lines = lines[:first_peak_idx]

    card_number = ""
    phase_name = ""
    for line in metadata_lines:
        if not card_number:
            m = PDF_CARD_PATTERN.search(line)
            if m:
                card_number = m.group(1).strip()
                continue
        if not phase_name and not PDF_CARD_PATTERN.search(line):
            phase_name = line[:100]

    if not card_number:
        card_number = f"Custom-{(phase_name or 'Unnamed')[:20]}"

    formula = ""
    if phase_name:
        m = FORMULA_PATTERN.search(phase_name)
        if m:
            cand = m.group(0)
            if any(c.isdigit() for c in cand) or sum(1 for c in cand if c.isupper()) >= 2:
                formula = cand

    # Detect schema from peak lines
    schema = _detect_schema(parsed_peak_parts)
    logger.info("Detected schema: %s from %d peak lines", schema, len(parsed_peak_parts))

    # Parse all peaks with schema
    peaks: list[dict[str, Any]] = []
    for parts in parsed_peak_parts:
        peak = _parse_peak_with_schema(parts, schema)
        if peak:
            peaks.append(peak)

    if len(peaks) < 3:
        raise ValueError(f"Too few peaks ({len(peaks)}). Need at least 3.")

    return {
        "card_number": card_number,
        "phase_name": phase_name or "Unknown phase",
        "formula": formula,
        "schema_detected": schema,
        "peaks": peaks,
        "n_peaks": len(peaks),
    }
