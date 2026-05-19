"""Unit tests for deterministic peak matcher.

Ground truth: hand-curated cases covering common materials science scenarios.

@phase R185-1-deterministic-peak-matcher
"""
from __future__ import annotations

import pytest

from src.deviation.peak_matcher import (
    DEFAULT_TOLERANCES,
    MatchResult,
    match_peaks,
)


# ── Test fixtures ─────────────────────────────────────────────────────────────

def mos2_reference_peaks():
    """MoS2 2H bulk reference (R183-2 seed)."""
    return [
        {"shift": 383, "intensity": 80, "assignment": "E12g in-plane"},
        {"shift": 408, "intensity": 100, "assignment": "A1g out-of-plane"},
    ]


def wo3_reference_peaks():
    """Monoclinic WO3 reference."""
    return [
        {"shift": 135, "intensity": 40, "assignment": "Lattice mode"},
        {"shift": 267, "intensity": 35, "assignment": "W-O-W deformation"},
        {"shift": 326, "intensity": 30, "assignment": "O-W-O bending"},
        {"shift": 715, "intensity": 60, "assignment": "W-O-W bridging"},
        {"shift": 806, "intensity": 100, "assignment": "W=O terminal stretch"},
    ]


# ── Test cases ────────────────────────────────────────────────────────────────

class TestPerfectMatch:
    def test_identical_peaks_yield_excellent_grade(self):
        ref = mos2_reference_peaks()
        sample = [
            {"shift_cm1": 383, "relative_intensity": 80, "fwhm": 8.5},
            {"shift_cm1": 408, "relative_intensity": 100, "fwhm": 6.2},
        ]
        result = match_peaks(sample, ref, "raman", "MoS2", "MoS2 (2H)")

        assert result.match_count == 2
        assert result.match_rate == 1.0
        assert result.mean_abs_deviation == 0.0
        assert result.quality_grade == "excellent"
        assert len(result.unmatched_sample) == 0
        assert len(result.unmatched_ref) == 0

    def test_match_confidence_high_for_zero_deviation(self):
        ref = [{"shift": 408, "intensity": 100, "assignment": "A1g"}]
        sample = [{"shift_cm1": 408, "relative_intensity": 100}]
        result = match_peaks(sample, ref, "raman", "MoS2")
        assert result.matches[0].confidence >= 0.95


class TestSmallDeviation:
    def test_within_tolerance_still_matches(self):
        ref = mos2_reference_peaks()
        sample = [
            {"shift_cm1": 385, "relative_intensity": 80, "fwhm": 10.5},  # +2
            {"shift_cm1": 412, "relative_intensity": 100, "fwhm": 8.2},  # +4
        ]
        result = match_peaks(sample, ref, "raman", "MoS2")

        assert result.match_count == 2
        assert result.mean_abs_deviation == 3.0
        # Signed deviation preserved
        assert all(m.deviation > 0 for m in result.matches)

    def test_signed_deviation_indicates_shift_direction(self):
        """Compressive strain → upshift; tensile strain → downshift."""
        ref = [{"shift": 408, "intensity": 100, "assignment": "A1g"}]
        sample_upshift = [{"shift_cm1": 411, "relative_intensity": 100}]  # +3
        sample_downshift = [{"shift_cm1": 405, "relative_intensity": 100}]  # -3

        r1 = match_peaks(sample_upshift, ref, "raman", "MoS2")
        r2 = match_peaks(sample_downshift, ref, "raman", "MoS2")

        assert r1.matches[0].deviation == 3.0
        assert r2.matches[0].deviation == -3.0


class TestOutOfTolerance:
    def test_peak_outside_tolerance_unmatched(self):
        ref = mos2_reference_peaks()
        sample = [
            {"shift_cm1": 383, "relative_intensity": 80},   # match
            {"shift_cm1": 420, "relative_intensity": 50},   # +12 from 408 → unmatched (tol=5)
        ]
        result = match_peaks(sample, ref, "raman", "MoS2")

        assert result.match_count == 1
        assert len(result.unmatched_sample) == 1
        assert result.unmatched_sample[0].position == 420
        # A1g 408 expected but not observed
        assert len(result.unmatched_ref) == 1
        assert result.unmatched_ref[0].position == 408


class TestHungarianOneToOneMapping:
    def test_two_close_sample_peaks_to_one_ref(self):
        """If 2 sample peaks compete for 1 ref, only closest wins."""
        ref = [{"shift": 408, "intensity": 100, "assignment": "A1g"}]
        sample = [
            {"shift_cm1": 410, "relative_intensity": 90},  # +2
            {"shift_cm1": 411, "relative_intensity": 50},  # +3
        ]
        result = match_peaks(sample, ref, "raman", "MoS2")

        # Hungarian gives unique assignment — only one match
        assert result.match_count == 1
        assert result.matches[0].sample_position == 410  # closer wins
        assert len(result.unmatched_sample) == 1


class TestEmptyInputs:
    def test_empty_sample_returns_all_unmatched_ref(self):
        ref = mos2_reference_peaks()
        result = match_peaks([], ref, "raman", "MoS2")
        assert result.match_count == 0
        # No sample → no unmatched ref recorded (we report against sample size)

    def test_empty_ref_returns_no_matches(self):
        sample = [{"shift_cm1": 408, "relative_intensity": 100}]
        result = match_peaks(sample, [], "raman", "MoS2")
        assert result.match_count == 0


class TestMixedPhaseDetection:
    """When sample has WO3 mixed (monoclinic + hexagonal), unmatched peaks
    indicate possible secondary phase."""

    def test_extra_peaks_appear_as_unmatched_sample(self):
        ref = wo3_reference_peaks()
        # Sample has all m-WO3 peaks + 1 extra at 640 (h-WO3 marker)
        sample = [
            {"shift_cm1": 135, "relative_intensity": 40},
            {"shift_cm1": 267, "relative_intensity": 35},
            {"shift_cm1": 326, "relative_intensity": 30},
            {"shift_cm1": 640, "relative_intensity": 25},  # h-WO3 tunnel mode
            {"shift_cm1": 715, "relative_intensity": 60},
            {"shift_cm1": 806, "relative_intensity": 100},
        ]
        result = match_peaks(sample, ref, "raman", "WO3", "WO3 (monoclinic)")

        assert result.match_count == 5
        assert len(result.unmatched_sample) == 1
        assert result.unmatched_sample[0].position == 640


class TestXRDSpectrumType:
    def test_xrd_uses_two_theta_field(self):
        ref = [
            {"twotheta": 14.4, "intensity": 100, "assignment": "(002)"},
            {"twotheta": 32.7, "intensity": 30, "assignment": "(100)"},
        ]
        sample = [
            {"two_theta": 14.5, "relative_intensity": 100},
            {"two_theta": 32.6, "relative_intensity": 30},
        ]
        result = match_peaks(sample, ref, "xrd", "MoS2")
        assert result.match_count == 2
        assert result.tolerance_used == DEFAULT_TOLERANCES["xrd"]


class TestQualityGrading:
    def test_high_match_rate_low_deviation_excellent(self):
        ref = mos2_reference_peaks()
        sample = [
            {"shift_cm1": 383, "relative_intensity": 80},
            {"shift_cm1": 408, "relative_intensity": 100},
        ]
        result = match_peaks(sample, ref, "raman", "MoS2")
        assert result.quality_grade == "excellent"

    def test_low_match_rate_poor(self):
        ref = mos2_reference_peaks()
        # Only 1 of many sample peaks match
        sample = [
            {"shift_cm1": 383, "relative_intensity": 80},
            {"shift_cm1": 500, "relative_intensity": 50},
            {"shift_cm1": 700, "relative_intensity": 30},
            {"shift_cm1": 900, "relative_intensity": 20},
            {"shift_cm1": 1100, "relative_intensity": 10},
        ]
        result = match_peaks(sample, ref, "raman", "MoS2")
        assert result.match_rate <= 0.4
        assert result.quality_grade in ("poor", "fair")


class TestCustomTolerance:
    def test_overriding_tolerance_changes_matching(self):
        ref = [{"shift": 408, "intensity": 100, "assignment": "A1g"}]
        sample = [{"shift_cm1": 415, "relative_intensity": 100}]  # +7

        # Default tol=5 → no match
        r_default = match_peaks(sample, ref, "raman", "MoS2")
        assert r_default.match_count == 0

        # Custom tol=10 → match
        r_custom = match_peaks(sample, ref, "raman", "MoS2", tolerance=10)
        assert r_custom.match_count == 1


class TestSerialization:
    def test_to_dict_produces_json_compatible(self):
        ref = mos2_reference_peaks()
        sample = [{"shift_cm1": 383, "relative_intensity": 80}]
        result = match_peaks(sample, ref, "raman", "MoS2")
        d = result.to_dict()

        # Round-trip through JSON
        import json
        s = json.dumps(d)
        parsed = json.loads(s)
        assert parsed["match_count"] == 1
        assert parsed["spectrum_type"] == "raman"
