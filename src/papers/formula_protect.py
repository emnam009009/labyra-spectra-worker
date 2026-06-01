"""Chemical-formula protection for translation (worker port of R270 TS).

Materials-science prose is dense with formulae (H2O, WO3, MoS2, Fe2O3) that the
model otherwise breaks (subscript split, "H 2 O") or transliterates. We mask any
whole-word token that parses ENTIRELY into periodic-table element symbols AND
looks chemical (>=2 element groups OR a subscript digit) BEFORE the model sees
it, then restore it verbatim. Ordinary words that happen to be element symbols
("In", "As", "He", "No", "Be") are left untouched.

Mirrors src/features/papers/lib/citation-protect.ts (R270) so on-demand and
pre-translate behave identically. Placeholder = U+27E6 'F' <idx> U+27E7 (math
white brackets) — almost never in prose, so the model passes it through; the
restore regex tolerates stray spaces the model may add.

@phase R225
"""
from __future__ import annotations

import re

# Periodic-table element symbols (1-118).
ELEMENTS = frozenset(
    {
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
        "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
        "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
    }
)

# Whole-word run of element-like groups (Capital + optional lowercase + digits).
# The \b...\b bounds mean formula-like fragments inside real words (the "Fi" in
# "Figure") never match — the trailing letter breaks the word boundary.
_CANDIDATE_RE = re.compile(r"\b(?:[A-Z][a-z]?\d*)+\b")
_GROUP_RE = re.compile(r"[A-Z][a-z]?\d*")
_GROUP_PARSE_RE = re.compile(r"^([A-Z][a-z]?)(\d*)$")
_RESTORE_RE = re.compile("\u27e6\\s*F(\\d+)\\s*\u27e7")


def is_chemical_formula(token: str) -> bool:
    """True when `token` parses fully into real element symbols AND looks chemical
    (>=2 element groups or at least one subscript digit)."""
    groups = _GROUP_RE.findall(token)
    if not groups:
        return False
    has_digit = False
    for g in groups:
        m = _GROUP_PARSE_RE.match(g)
        if not m or m.group(1) not in ELEMENTS:
            return False
        if m.group(2):
            has_digit = True
    return len(groups) >= 2 or has_digit


def protect_formulae(text: str) -> tuple[str, list[str]]:
    """Mask chemical formulae with U+27E6 F<idx> U+27E7. Returns (masked, formulae)."""
    formulae: list[str] = []

    def repl(m: re.Match[str]) -> str:
        tok = m.group(0)
        if not is_chemical_formula(tok):
            return tok
        idx = len(formulae)
        formulae.append(tok)
        return f"\u27e6F{idx}\u27e7"

    return _CANDIDATE_RE.sub(repl, text), formulae


def restore_formulae(text: str, formulae: list[str]) -> str:
    """Restore masked formulae verbatim. Tolerant of spaces the model may insert
    (e.g. U+27E6 F 1 U+27E7). Orphaned placeholders are dropped."""
    if not formulae:
        return text

    def repl(m: re.Match[str]) -> str:
        i = int(m.group(1))
        return formulae[i] if 0 <= i < len(formulae) else ""

    return _RESTORE_RE.sub(repl, text)
