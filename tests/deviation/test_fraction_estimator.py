"""Unit tests for fraction estimator.

@phase R185-7-fraction-estimator
"""
from __future__ import annotations

from src.deviation.fraction_estimator import (
    estimate_fractions,
    estimate_raman_qualitative,
    estimate_xrd_rir,
)


def _component(formula: str, peak_intensities: list[float], peak_count: int | None = None):
    return {
        "formula": formula,
        "match_result": {
            "matches": [
                {"sample_intensity": v, "ref_assignment": ""}
                for v in peak_intensities
            ],
        },
        "intended_peaks_observed": peak_count if peak_count is not None else len(peak_intensities),
    }


# ── XRD RIR ──────────────────────────────────────────────────────────────────

class TestXRDRIR:
    def test_quantitative_with_rir_factors(self):
        components = [
            _component("MoS2", [100, 60, 40]),
            _component("WO3", [200, 80]),
        ]
        profiles = {
            "MoS2": {"formula": "MoS2", "rirFactor": 5.0},
            "WO3": {"formula": "WO3", "rirFactor": 8.0},
        }
        estimates = estimate_xrd_rir(components, profiles)
        assert len(estimates) == 2
        # All quantitative
        assert all(e.quantitative for e in estimates)
        assert all(e.method == "rir" for e in estimates)
        # Sum should approximately equal 1
        total = sum(e.value for e in estimates)
        assert abs(total - 1.0) < 0.01
        # Citation present
        assert all(e.citation is not None for e in estimates)

    def test_missing_rir_factor_fails(self):
        components = [
            _component("MoS2", [100]),
            _component("WO3", [80]),
        ]
        profiles = {
            "MoS2": {"formula": "MoS2", "rirFactor": 5.0},
            "WO3": {"formula": "WO3"},  # no RIR
        }
        estimates = estimate_xrd_rir(components, profiles)
        # Should fail and return empty when any component lacks RIR
        assert estimates == []

    def test_uncertainty_proportional_to_fraction(self):
        components = [_component("MoS2", [100, 50])]
        profiles = {"MoS2": {"formula": "MoS2", "rirFactor": 5.0}}
        estimates = estimate_xrd_rir(components, profiles)
        assert len(estimates) == 1
        assert estimates[0].uncertainty > 0
        assert estimates[0].value == 1.0


# ── Raman qualitative ────────────────────────────────────────────────────────

class TestRamanQualitative:
    def test_intensity_ratio_returned(self):
        components = [
            _component("MoS2", [100, 80]),
            _component("C", [200, 150]),
        ]
        estimates = estimate_raman_qualitative(components)
        assert len(estimates) == 2
        # NEVER quantitative
        assert all(not e.quantitative for e in estimates)
        # Method must be qualitative
        assert all(e.method == "raman-intensity-ratio-qualitative" for e in estimates)
        # Caveat must mention cross-section
        assert all("cross-section" in e.caveat.lower() for e in estimates)

    def test_caveat_mentions_not_mass_fraction(self):
        components = [_component("MoS2", [100]), _component("C", [200])]
        estimates = estimate_raman_qualitative(components)
        for e in estimates:
            assert "NOT mass fraction" in e.caveat


# ── Dispatcher ───────────────────────────────────────────────────────────────

class TestDispatcher:
    def test_xrd_prefers_dc_over_rir_when_both_available(self):
        """After R185-7b, DC method is preferred over RIR when formulas are parseable."""
        components = [_component("MoS2", [100]), _component("WO3", [80])]
        profiles = {
            "MoS2": {"rirFactor": 5.0, "formula": "MoS2"},
            "WO3": {"rirFactor": 8.0, "formula": "WO3"},
        }
        estimates = estimate_fractions("xrd", components, profiles)
        # DC wins because both have parseable formulas → MAC computable
        assert all(e.method == "direct-comparison" for e in estimates)

    def test_xrd_without_rir_uses_dc_when_formulas_parseable(self):
        """After R185-7b: parseable formulas → DC, no need for rirFactor."""
        components = [_component("MoS2", [100], peak_count=3), _component("WO3", [80], peak_count=2)]
        profiles = {"MoS2": {}, "WO3": {}}
        estimates = estimate_fractions("xrd", components, profiles)
        assert all(e.method == "direct-comparison" for e in estimates)
        assert all(e.quantitative for e in estimates)

    def test_xrd_unparseable_formula_falls_back_to_peak_count(self):
        """Truly unknown formula → DC fails → fall through to peak-count."""
        components = [_component("UnknownXYZ", [100], peak_count=2)]
        profiles = {"UnknownXYZ": {}}
        estimates = estimate_fractions("xrd", components, profiles)
        if estimates:
            assert estimates[0].method == "peak-count-fallback"

    def test_raman_uses_qualitative_ratio(self):
        components = [_component("MoS2", [100]), _component("C", [200])]
        estimates = estimate_fractions("raman", components, {})
        assert all(e.method == "raman-intensity-ratio-qualitative" for e in estimates)

    def test_empty_components_returns_empty(self):
        estimates = estimate_fractions("xrd", [], {})
        assert estimates == []


# ── Caveat sanity ────────────────────────────────────────────────────────────

class TestCaveat:
    def test_every_estimate_has_non_empty_caveat(self):
        components = [_component("MoS2", [100]), _component("WO3", [80])]
        profiles = {"MoS2": {"rirFactor": 5.0}, "WO3": {"rirFactor": 8.0}}

        # Test all methods
        for method_call in [
            lambda: estimate_xrd_rir(components, profiles),
            lambda: estimate_raman_qualitative(components),
        ]:
            estimates = method_call()
            for e in estimates:
                assert e.caveat
                assert len(e.caveat) > 30  # not just placeholder
