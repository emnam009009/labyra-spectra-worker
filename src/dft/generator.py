"""
generator.py — render an ordered Quantum ESPRESSO input from a structure + params
via Jinja2. Solves the "messy input" problem: params in any order → canonical
namelist order (CONTROL→SYSTEM→ELECTRONS→IONS→CELL) + cards
(ATOMIC_SPECIES→CELL_PARAMETERS→ATOMIC_POSITIONS→K_POINTS→HUBBARD). The structure
block reuses the verified ibrav/celldm from structure.py.

@phase R272w-b (DFT P0 — input generator)
"""
from __future__ import annotations

import os
from typing import Any

from jinja2 import Environment, FileSystemLoader

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

# pw.x covers relax/scf/nscf/bands; post-processing executables (bands.x/dos.x/
# projwfc.x/pp.x) get their own small templates in P0-2b.
_TEMPLATE_BY_CALC: dict[str, str] = {
    "vc-relax": "pw.in.j2",
    "relax": "pw.in.j2",
    "scf": "pw.in.j2",
    "nscf": "pw.in.j2",
    "bands": "pw.in.j2",
}


def qe_bool(v: Any) -> str:
    """Python bool (or Jinja Undefined) → Fortran logical."""
    return ".true." if v else ".false."


def qe_sci(v: Any) -> str:
    """Float → Fortran D-notation, e.g. 1e-9 → '1.0d-9'."""
    mantissa, exp = f"{float(v):.1e}".split("e")
    return f"{mantissa}d{int(exp)}"


def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        keep_trailing_newline=True,
    )
    env.filters["qe_bool"] = qe_bool
    env.filters["qe_sci"] = qe_sci
    return env


def generate_pw_input(
    structure: dict[str, Any],
    params: dict[str, Any],
    *,
    prefix: str,
    ecutwfc: float,
    ecutrho: float,
    functional: str = "pbe",
    hubbard: list[dict[str, Any]] | None = None,
    outdir: str | None = None,
) -> str:
    """Render a pw.x input (.in) for one calculation.

    structure : DftStructure dict from structure.py (to_dft_structure).
    params    : per-calc params — must include `calculation` and `kPoints`;
                optional tstress/tprnfor/nbnd/occupations/convThr/... (the template
                supplies safe defaults). Pass Fortran-bool fields as Python bools.
    """
    calc = params.get("calculation")
    template_name = _TEMPLATE_BY_CALC.get(calc)
    if template_name is None:
        raise ValueError(f"unsupported pw.x calculation: {calc!r}")
    wf = {
        "prefix": prefix,
        "ecutwfc": ecutwfc,
        "ecutrho": ecutrho,
        "functional": functional,
        "hubbard": hubbard or [],
    }
    unit = {"outdir": outdir or f"./outdir_{calc}"}
    return _env().get_template(template_name).render(wf=wf, unit=unit, s=structure, p=params)
