"""Unit tests for crystallinity classifier.

@phase R185-5-crystallinity-classifier
"""
from __future__ import annotations

from src.deviation.crystallinity import (
    adaptive_tolerance,
    classify_crystallinity,
)


class TestBulkClassification:
    def test_narrow_fwhm_yields_bulk(self):
        sample = [
            {"shift_cm1": 383, "intensity": 80, "fwhm": 8.5},
            {"shift_cm1": 408, "intensity": 100, "fwhm": 7.2},
        ]
        ref = [
            {"shift": 383, "intensity": 80, "fwhm": 8.0, "assignment": "E12g"},
            {"shift": 408, "intensity": 100, "fwhm": 7.0, "assignment": "A1g"},
        ]
        matches = [
            {"sample_position": 383, "ref_position": 383, "deviation": 0},
            {"sample_position": 408, "ref_position": 408, "deviation": 0},
        ]

        result = classify_crystallinity("raman", {"peaks": sample}, sample, ref, matches)
        assert result.classification == "bulk"
        assert result.confidence > 0.7
        assert result.tolerance_factor == 1.0


class TestNanocrystalline:
    def test_broadened_fwhm_yields_nano(self):
        sample = [
            {"shift_cm1": 144, "intensity": 100, "fwhm": 16.0},  # 2x ref FWHM
            {"shift_cm1": 639, "intensity": 40, "fwhm": 14.0},
        ]
        ref = [
            {"shift": 144, "intensity": 100, "fwhm": 8.0, "assignment": "Eg"},
            {"shift": 639, "intensity": 40, "fwhm": 7.0, "assignment": "Eg"},
        ]
        matches = [
            {"sample_position": 144, "ref_position": 144, "deviation": 0},
            {"sample_position": 639, "ref_position": 639, "deviation": 0},
        ]
        result = classify_crystallinity("raman", {"peaks": sample}, sample, ref, matches)
        assert result.classification == "nanocrystalline"
        assert result.size_estimate is not None
        assert result.size_estimate.method == "phonon-confinement"
        assert result.tolerance_factor > 1.0


class TestAmorphous:
    def test_very_broad_fwhm_yields_amorphous(self):
        sample = [
            {"shift_cm1": 144, "intensity": 100, "fwhm": 40.0},  # 5x
            {"shift_cm1": 639, "intensity": 40, "fwhm": 35.0},
        ]
        ref = [
            {"shift": 144, "intensity": 100, "fwhm": 8.0, "assignment": "Eg"},
            {"shift": 639, "intensity": 40, "fwhm": 7.0, "assignment": "Eg"},
        ]
        matches = [
            {"sample_position": 144, "ref_position": 144, "deviation": 0},
            {"sample_position": 639, "ref_position": 639, "deviation": 0},
        ]
        result = classify_crystallinity("raman", {"peaks": sample}, sample, ref, matches)
        assert result.classification == "amorphous"
        assert result.tolerance_factor == 3.0


class TestBackgroundCorroboration:
    def test_high_background_boosts_amorphous_confidence(self):
        sample = [{"shift_cm1": 144, "intensity": 100, "fwhm": 40.0}]
        ref = [{"shift": 144, "intensity": 100, "fwhm": 8.0, "assignment": "Eg"}]
        parsed = {
            "peaks": sample,
            "curve": {"y": [50, 55, 60, 58, 52, 100, 95, 60, 55, 50, 52, 58, 60]},  # high baseline, >10 points
        }
        result = classify_crystallinity(
            "raman", parsed, sample, ref,
            matches=[{"sample_position": 144, "ref_position": 144, "deviation": 0}],
        )
        assert result.classification == "amorphous"
        # Background ratio is 50/100 = 0.5 > 0.3 → boosts confidence
        assert result.signals.background_ratio is not None


class TestSizeEstimate:
    def test_pcm_size_within_reasonable_range(self):
        sample = [{"shift_cm1": 144, "intensity": 100, "fwhm": 20.0}]
        ref = [{"shift": 144, "intensity": 100, "fwhm": 8.0, "assignment": "Eg"}]
        result = classify_crystallinity(
            "raman", {"peaks": sample}, sample, ref,
            matches=[{"sample_position": 144, "ref_position": 144, "deviation": 0}],
        )
        assert result.size_estimate is not None
        # FWHM ratio ~2.5 → PCM size ~5 nm
        assert 1.0 < result.size_estimate.value_nm < 30.0
        assert result.size_estimate.citation is not None
        assert "PhysRevB" in result.size_estimate.citation.doi


class TestAdaptiveTolerance:
    def test_factor_per_classification(self):
        assert adaptive_tolerance(5.0, "bulk") == 5.0
        assert adaptive_tolerance(5.0, "nanocrystalline") == 7.5
        assert adaptive_tolerance(5.0, "amorphous") == 15.0
        assert adaptive_tolerance(5.0, "unknown") == 5.0


class TestNoReference:
    def test_no_ref_data_returns_unknown(self):
        sample = [{"shift_cm1": 144, "intensity": 100, "fwhm": 10.0}]
        result = classify_crystallinity("raman", {"peaks": sample}, sample, [], [])
        # Without reference fwhm, classifier falls back to unknown
        assert result.classification in ("unknown", "amorphous")
