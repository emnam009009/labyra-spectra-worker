"""
CV analysis tests (R254).

Redox-couple descriptors from a cyclic voltammogram: Epa/Epc, peak separation
dEp, formal potential E0', peak current ratio, and reversibility classification.
Synthetic Gaussian redox peaks on forward/reverse sweeps.
"""

from __future__ import annotations

import numpy as np

from src.parsers.cv import parse_cv


def _peak(e: np.ndarray, center: float, amp: float, w: float) -> np.ndarray:
    return amp * np.exp(-((e - center) ** 2) / (2 * w**2))


def _cv_text(e0: float = 0.2, dep_v: float = 0.059, ratio: float = 1.0) -> str:
    e_fwd = np.linspace(-0.2, 0.6, 200)
    e_rev = np.linspace(0.6, -0.2, 200)
    i_fwd = _peak(e_fwd, e0 + dep_v / 2, 1.0, 0.04) + 0.05
    i_rev = -_peak(e_rev, e0 - dep_v / 2, ratio, 0.04) - 0.05
    e = np.concatenate([e_fwd, e_rev])
    i = np.concatenate([i_fwd, i_rev])
    return "\n".join(f"{a:.5f},{b:.6f}" for a, b in zip(e, i, strict=False))


# --- core descriptors -------------------------------------------------------

def test_peak_separation() -> None:
    a = parse_cv(_cv_text(dep_v=0.059), n_electrons=1)["analysis"]
    assert abs(a["dEp_mV"] - 59.0) < 5.0


def test_formal_potential() -> None:
    a = parse_cv(_cv_text(e0=0.20), n_electrons=1)["analysis"]
    assert abs(a["E0_prime_V"] - 0.20) < 0.005


def test_peak_current_ratio() -> None:
    a = parse_cv(_cv_text(ratio=1.0), n_electrons=1)["analysis"]
    assert abs(a["peak_current_ratio"] - 1.0) < 0.1


def test_ideal_dep_scales_with_n() -> None:
    a = parse_cv(_cv_text(), n_electrons=2)["analysis"]
    assert abs(a["dEp_ideal_mV"] - 29.5) < 0.1  # 59/2


# --- reversibility classification -------------------------------------------

def test_reversible_classification() -> None:
    a = parse_cv(_cv_text(dep_v=0.059, ratio=1.0), n_electrons=1)["analysis"]
    assert "reversible-like" in a["reversibility"]


def test_quasi_reversible_classification() -> None:
    a = parse_cv(_cv_text(dep_v=0.150, ratio=0.7), n_electrons=1)["analysis"]
    assert "quasi-reversible" in a["reversibility"]


def test_irreversible_classification() -> None:
    a = parse_cv(_cv_text(dep_v=0.300, ratio=0.5), n_electrons=1)["analysis"]
    assert "irreversible" in a["reversibility"]


# --- guards -----------------------------------------------------------------

def test_scan_rate_note_without_rate() -> None:
    r = parse_cv(_cv_text(), n_electrons=1)  # no scan_rate
    assert any("scan-rate series" in n.lower() or "randles" in n.lower() for n in r["notes"])


def test_provisional_reversibility_note() -> None:
    r = parse_cv(_cv_text(), n_electrons=1)
    assert any("provisional" in n.lower() for n in r["notes"])
