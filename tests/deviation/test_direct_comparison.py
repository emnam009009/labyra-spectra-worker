"""Unit tests for Direct Comparison (Klug-Alexander) fraction estimation.

@phase R185-7b-direct-comparison-method
"""
from __future__ import annotations

import pytest

from src.deviation.fraction_estimator import (
    estimate_fractions,
    estimate_xrd_direct_comparison,
)
from src.deviation.mass_absorption import compound_mac, get_mac_for_anode


def _component(formula: str, peak_intensities: list[float]):
    return {
        "formula": formula,
        "match_result": {
            "matches": [
                {"sample_intensity": v, "ref_assignment": ""}
                for v in peak_intensities
            ],
        },
        "intended_peaks_observed": len(peak_intensities),
    }


# ── Mass absorption coefficient lookup ───────────────────────────────────────

class TestMassAbsorption:
    def test_pure_element_mac(self):
        assert compound_mac("Fe", "Cu") == pytest.approx(308.0, rel=0.01)
        assert compound_mac("Cu", "Cu") == pytest.approx(52.7, rel=0.01)

    def test_compound_mac_weighted(self):
        # MoS2: Mo (mass=95.94) + 2 S (mass=32.06)
        # Total = 160.06; w_Mo = 95.94/160.06 = 0.599; w_S = 0.401
        # mu/rho = 0.599 * 25.6 + 0.401 * 93.3 = 15.3 + 37.4 = 52.7
        mac = compound_mac("MoS2", "Cu")
        assert mac == pytest.approx(52.7, rel=0.05)

    def test_water_mac(self):
        # H2O: w_H = 2*1.008/18.015 = 0.112; w_O = 0.888
        # mu/rho = 0.112 * 0.392 + 0.888 * 11.5 = 10.3
        mac = compound_mac("H2O", "Cu")
        assert mac == pytest.approx(10.3, rel=0.05)

    def test_unknown_element_returns_none(self):
        # Uup (ununpentium) not in our table
        mac = compound_mac("Uup", "Cu")
        assert mac is None

    def test_invalid_formula_returns_none(self):
        mac = compound_mac("XYZ123!@#", "Cu")
        assert mac is None


# ── Direct Comparison method ─────────────────────────────────────────────────

class TestDirectComparison:
    def test_two_phase_composite_quantitative(self):
        # MoS2 + WO3 — both have well-defined MACs
        components = [
            _component("MoS2", [100, 60, 40]),
            _component("WO3", [200, 80]),
        ]
        profiles = {"MoS2": {"formula": "MoS2"}, "WO3": {"formula": "WO3"}}

        estimates = estimate_xrd_direct_comparison(components, profiles)
        assert len(estimates) == 2
        assert all(e.quantitative for e in estimates)
        assert all(e.method == "direct-comparison" for e in estimates)
        # Sum ~ 1.0
        total = sum(e.value for e in estimates)
        assert abs(total - 1.0) < 0.01
        # Citation present
        assert all(e.citation is not None for e in estimates)
        assert all("Klug" in e.citation.journal or "Klug" in e.citation.title for e in estimates)

    def test_no_match_intensity_returns_empty(self):
        components = [_component("MoS2", [])]
        profiles = {"MoS2": {}}
        estimates = estimate_xrd_direct_comparison(components, profiles)
        assert estimates == []

    def test_uncertainty_lower_than_rir(self):
        components = [_component("MoS2", [100]), _component("WO3", [80])]
        profiles = {"MoS2": {}, "WO3": {}}

        dc_estimates = estimate_xrd_direct_comparison(components, profiles)
        # DC uncertainty should be ~5% vs RIR ~8%
        assert all(e.uncertainty <= 0.06 for e in dc_estimates)

    def test_caveat_mentions_mac_value(self):
        components = [_component("MoS2", [100]), _component("C", [200])]
        profiles = {"MoS2": {}, "C": {}}
        estimates = estimate_xrd_direct_comparison(components, profiles)
        for e in estimates:
            assert "mu/rho" in e.caveat or "mass absorption" in e.caveat.lower() or "Cu Kalpha" in e.caveat


# ── Dispatcher routing ───────────────────────────────────────────────────────

class TestDispatcherPrefersDC:
    def test_xrd_prefers_dc_over_rir(self):
        """When both DC and RIR can run, DC should be selected (more accurate)."""
        components = [_component("MoS2", [100]), _component("WO3", [80])]
        profiles = {
            "MoS2": {"rirFactor": 5.0},  # has RIR
            "WO3": {"rirFactor": 8.0},   # has RIR
        }
        estimates = estimate_fractions("xrd", components, profiles)
        # DC should win because both phases have parseable formulas
        assert all(e.method == "direct-comparison" for e in estimates)

    def test_xrd_falls_back_to_peak_count_only_if_both_fail(self):
        # Use a formula that lacks element MAC data → DC fails
        # Also no rirFactor → RIR fails
        # Should fall to peak-count
        components = [_component("UnknownElementXyz", [100])]
        profiles = {"UnknownElementXyz": {}}
        estimates = estimate_fractions("xrd", components, profiles)
        # Either empty or peak-count
        if estimates:
            assert estimates[0].method == "peak-count-fallback"


# ── Anode awareness ──────────────────────────────────────────────────────────

class TestAnodeAware:
    def test_mo_anode_different_macs(self):
        # Mo Kalpha has much smaller MAC than Cu Kalpha
        mac_cu = get_mac_for_anode("Fe", "Cu")
        mac_mo = get_mac_for_anode("Fe", "Mo")
        assert mac_cu > mac_mo > 0
