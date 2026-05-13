"""Parse and normalize chemical formulas."""

from __future__ import annotations

import re

# Element symbols (subset of periodic table common in materials science)
ELEMENT_PATTERN = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def parse_formula(formula: str) -> dict[str, float]:
    """Parse 'WO3' → {'W': 1, 'O': 3}, 'Fe2O3' → {'Fe': 2, 'O': 3}.

    Handles:
      - Simple stoichiometric (WO3, TiO2, Fe2O3)
      - Decimal (W0.5O1.5)
      - Implicit 1 (NaCl → {Na:1, Cl:1})

    Does NOT handle: parentheses, hydrates, charges.
    """
    if not formula:
        return {}

    # Strip whitespace and common prefixes
    f = formula.strip()
    f = re.sub(r"[\s\-\+]", "", f)

    counts: dict[str, float] = {}
    matches = ELEMENT_PATTERN.findall(f)
    if not matches:
        return {}

    for element, count_str in matches:
        if not element:
            continue
        count = float(count_str) if count_str else 1.0
        counts[element] = counts.get(element, 0.0) + count

    return counts


def extract_formula_from_label(label: str) -> str | None:
    """Best-effort extract formula from sample label.

    Examples:
      'WO3-100C-1h' → 'WO3'
      'sample-A_TiO2_anneal_500C' → 'TiO2'
      'Fe2O3 nanoparticles' → 'Fe2O3'
      'unknown sample' → None
    """
    if not label:
        return None

    # Try common formula patterns
    patterns = [
        # Stoichiometric: 2+ uppercase letters/element
        r"\b([A-Z][a-z]?\d*[A-Z][a-z]?\d*[A-Z]?[a-z]?\d*)\b",
        # Pure element with subscript
        r"\b([A-Z][a-z]?\d+)\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, label):
            candidate = match.group(1)
            parsed = parse_formula(candidate)
            # Must have at least 2 elements OR an oxide-like single-element-with-number
            if len(parsed) >= 2:
                return candidate
            if len(parsed) == 1 and any(c > 1 for c in parsed.values()):
                return candidate

    return None


def normalize_formula(formula: str) -> str:
    """Convert to Hill-system canonical (C first if present, then H, then alphabetical).

    For materials science: alphabetical (C and H aren't special for inorganics).
    'O3W' → 'WO3' (re-order alphabetical with counts).
    """
    parsed = parse_formula(formula)
    if not parsed:
        return formula
    parts = []
    for el in sorted(parsed.keys()):
        c = parsed[el]
        if c == 1.0:
            parts.append(el)
        elif c == int(c):
            parts.append(f"{el}{int(c)}")
        else:
            parts.append(f"{el}{c}")
    return "".join(parts)


def elements_only(formula: str) -> list[str]:
    """Return list of unique elements in formula, sorted alphabetically.

    Used for COD element search (el1, el2, ...).
    """
    parsed = parse_formula(formula)
    return sorted(parsed.keys())
