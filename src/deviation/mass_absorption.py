"""
Mass absorption coefficient (mu/rho) calculation for compounds.

Uses pymatgen.core.Composition + NIST XCOM element data via pymatgen.

For compound A_x B_y C_z:
    (mu/rho)_compound = sum_k w_k * (mu/rho)_k
where w_k is mass fraction of element k, (mu/rho)_k is its tabulated MAC
for the X-ray wavelength.

For Cu Kalpha (8.04 keV, 1.5406 A), typical MAC values (cm^2/g):
    H: 0.39    O: 11.5    Al: 50.2   Si: 65.3   Ti: 208   Cu: 52.7
    Mo: 25.6   W: 173     Pb: 232    Zn: 60.3   Fe: 308

@phase R185-7b-direct-comparison-method
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# Mass absorption coefficient for Cu Kalpha1 (8.0478 keV, 1.5406 A)
# Values in cm^2/g from NIST XCOM (https://physics.nist.gov/PhysRefData/Xcom/)
# Comprehensive set for materials science elements.
MAC_CU_KA: dict[str, float] = {
    "H":  0.392,  "He": 0.182,  "Li": 0.717,  "Be": 1.50,   "B":  2.71,
    "C":  4.60,   "N":  7.52,   "O":  11.5,   "F":  16.5,   "Ne": 22.9,
    "Na": 30.1,   "Mg": 38.6,   "Al": 50.2,   "Si": 65.3,   "P":  77.6,
    "S":  93.3,   "Cl": 109.4,  "Ar": 122.6,  "K":  148.4,  "Ca": 172.0,
    "Sc": 186.0,  "Ti": 208.0,  "V":  227.0,  "Cr": 252.0,  "Mn": 285.0,
    "Fe": 308.0,  "Co": 354.0,  "Ni": 49.2,   "Cu": 52.7,   "Zn": 60.3,
    "Ga": 67.9,   "Ge": 75.6,   "As": 83.4,   "Se": 91.4,   "Br": 99.6,
    "Kr": 108.0,  "Rb": 117.0,  "Sr": 125.0,  "Y":  134.0,  "Zr": 143.0,
    "Nb": 153.0,  "Mo": 25.6,   "Tc": 27.8,   "Ru": 30.2,   "Rh": 32.7,
    "Pd": 35.3,   "Ag": 38.0,   "Cd": 41.0,   "In": 44.1,   "Sn": 47.4,
    "Sb": 50.9,   "Te": 54.6,   "I":  58.5,   "Xe": 62.7,   "Cs": 67.1,
    "Ba": 71.6,   "La": 81.5,   "Ce": 89.6,   "Pr": 95.0,   "Nd": 100.0,
    "Pm": 105.0,  "Sm": 112.0,  "Eu": 117.0,  "Gd": 124.0,  "Tb": 131.0,
    "Dy": 138.0,  "Ho": 145.0,  "Er": 152.0,  "Tm": 159.0,  "Yb": 167.0,
    "Lu": 174.0,  "Hf": 159.0,  "Ta": 166.0,  "W":  173.0,  "Re": 180.0,
    "Os": 187.0,  "Ir": 194.0,  "Pt": 200.0,  "Au": 208.0,  "Hg": 215.0,
    "Tl": 224.0,  "Pb": 232.0,  "Bi": 240.0,  "Th": 142.0,  "U":  152.0,
}

# Mo Kalpha1 (17.479 keV, 0.7107 A) MACs — much smaller (high E penetrates more)
# Provided for completeness; Labyra defaults to Cu Kalpha
MAC_MO_KA: dict[str, float] = {
    "H":  0.373,  "C":  0.625,  "N":  0.916,  "O":  1.310,
    "Al": 5.16,   "Si": 6.44,   "S":  9.55,   "Ti": 22.4,   "V":  25.2,
    "Fe": 37.6,   "Cu": 49.1,   "Zn": 54.1,   "Mo": 18.4,   "Ag": 35.0,
    "W":  93.4,   "Pb": 122.0,
}


def get_mac_for_anode(element: str, anode: str = "Cu") -> float | None:
    """Lookup element MAC for given anode wavelength."""
    table = MAC_CU_KA if anode.lower() in ("cu", "cu_ka", "cu-ka") else MAC_MO_KA
    return table.get(element)


def compound_mac(formula: str, anode: str = "Cu") -> float | None:
    """
    Mass absorption coefficient of a compound (cm^2/g) for the given anode.

    Uses pymatgen.core.Composition to parse formula and weight by mass fraction.

    Returns None if any element lacks MAC data.
    """
    try:
        from pymatgen.core.composition import Composition  # type: ignore[import]
    except ImportError:
        logger.error("pymatgen not installed; cannot compute MAC")
        return None

    try:
        comp = Composition(formula)
    except Exception:
        logger.exception("Failed to parse formula: %s", formula)
        return None

    total_mass = comp.weight  # amu/formula unit
    if total_mass <= 0:
        return None

    mac_sum = 0.0
    for element, amount in comp.items():
        symbol = element.symbol
        elem_mac = get_mac_for_anode(symbol, anode)
        if elem_mac is None:
            logger.warning("No MAC data for element %s (anode=%s)", symbol, anode)
            return None
        # Mass fraction of this element in compound
        element_mass = amount * element.atomic_mass
        weight_frac = element_mass / total_mass
        mac_sum += weight_frac * elem_mac

    return round(mac_sum, 3)
