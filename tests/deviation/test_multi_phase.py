"""Unit tests for multi-phase greedy matcher.

@phase R185-4-multi-phase-matcher
"""
from __future__ import annotations

from src.deviation.multi_phase import (
    ComponentDeclaration,
    match_multi_phase,
)


def _mos2_profile():
    return {
        "formula": "MoS2",
        "commonNames": ["molybdenum disulfide"],
        "spectralSignatures": {
            "raman": {
                "peaks": [
                    {"shift": 383, "intensity": 80, "assignment": "E12g"},
                    {"shift": 408, "intensity": 100, "assignment": "A1g"},
                ],
            },
        },
    }


def _carbon_profile():
    return {
        "formula": "C",
        "commonNames": ["graphite"],
        "spectralSignatures": {
            "raman": {
                "peaks": [
                    {"shift": 1350, "intensity": 50, "assignment": "D-band"},
                    {"shift": 1580, "intensity": 100, "assignment": "G-band"},
                    {"shift": 2700, "intensity": 60, "assignment": "2D-band"},
                ],
            },
        },
    }


def _wo3_profile():
    return {
        "formula": "WO3",
        "commonNames": ["monoclinic WO3"],
        "spectralSignatures": {
            "raman": {
                "peaks": [
                    {"shift": 715, "intensity": 60, "assignment": "W-O-W"},
                    {"shift": 806, "intensity": 100, "assignment": "W=O"},
                ],
            },
        },
    }


def _loader(profiles: dict[str, dict]):
    def load(formula: str):
        return profiles.get(formula)
    return load


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestMoS2rGOComposite:
    """MoS2 (matrix) + C (support, e.g. rGO) heterostructure."""

    def test_both_phases_detected(self):
        sample = [
            {"shift_cm1": 383, "relative_intensity": 80, "fwhm": 8},
            {"shift_cm1": 408, "relative_intensity": 100, "fwhm": 7},
            {"shift_cm1": 1350, "relative_intensity": 50, "fwhm": 30},
            {"shift_cm1": 1580, "relative_intensity": 90, "fwhm": 25},
        ]
        components = [
            ComponentDeclaration("MoS2", "matrix", 0.7),
            ComponentDeclaration("C", "support", 0.3),
        ]
        loader = _loader({"MoS2": _mos2_profile(), "C": _carbon_profile()})

        result = match_multi_phase(sample, components, loader, "raman")

        assert len(result.components) == 2
        # MoS2 should be processed first (matrix > support)
        assert result.components[0].formula == "MoS2"
        assert result.components[0].intended_peaks_observed == 2
        # C should detect 2 of 3 (D + G, no 2D in this sample)
        assert result.components[1].formula == "C"
        assert result.components[1].intended_peaks_observed == 2
        # Both phases fully accounted for
        assert len(result.unassigned_peaks) == 0
        assert result.overall_match_rate == 1.0

    def test_greedy_does_not_double_assign(self):
        """If MoS2 takes peak 383, C cannot also claim 383."""
        sample = [
            {"shift_cm1": 383, "relative_intensity": 80},
            {"shift_cm1": 408, "relative_intensity": 100},
        ]
        components = [
            ComponentDeclaration("MoS2", "matrix", 0.7),
            ComponentDeclaration("C", "support", 0.3),
        ]
        loader = _loader({"MoS2": _mos2_profile(), "C": _carbon_profile()})

        result = match_multi_phase(sample, components, loader, "raman")

        # MoS2 gets both
        assert result.components[0].intended_peaks_observed == 2
        # C gets nothing (no carbon peaks in sample)
        assert result.components[1].intended_peaks_observed == 0
        # C flagged as intended-but-not-observed
        assert "C" in result.intended_but_not_observed


class TestIntendedButNotObserved:
    def test_declared_phase_missing_from_sample(self):
        # Sample only has MoS2 peaks; user declared MoS2 + WO3 composite
        sample = [
            {"shift_cm1": 383, "relative_intensity": 80},
            {"shift_cm1": 408, "relative_intensity": 100},
        ]
        components = [
            ComponentDeclaration("MoS2", "matrix", 0.7),
            ComponentDeclaration("WO3", "filler", 0.3),
        ]
        loader = _loader({"MoS2": _mos2_profile(), "WO3": _wo3_profile()})

        result = match_multi_phase(sample, components, loader, "raman")

        # WO3 declared but not observed
        assert "WO3" in result.intended_but_not_observed
        wo3 = [c for c in result.components if c.formula == "WO3"][0]
        assert wo3.intent_coverage < 0.3


class TestUnassignedPeaks:
    def test_unexpected_peaks_flagged(self):
        # Sample has MoS2 + an unexpected strong peak at 520 (Si substrate)
        sample = [
            {"shift_cm1": 383, "relative_intensity": 80},
            {"shift_cm1": 408, "relative_intensity": 100},
            {"shift_cm1": 520, "relative_intensity": 70},  # unexpected
        ]
        components = [ComponentDeclaration("MoS2", "matrix", 1.0)]
        loader = _loader({"MoS2": _mos2_profile()})

        result = match_multi_phase(sample, components, loader, "raman")

        assert len(result.unassigned_peaks) == 1
        assert result.unassigned_peaks[0]["position"] == 520


class TestRoleOrdering:
    def test_matrix_processed_before_support(self):
        # Setup where ordering matters: peak at 383 could be claimed by either
        # if both refs had it. Verify matrix (higher weight) wins.
        sample = [{"shift_cm1": 383, "relative_intensity": 80}]
        # Both profiles have 383 for demo
        profile_a = {
            "formula": "A",
            "commonNames": ["fake A"],
            "spectralSignatures": {"raman": {"peaks": [{"shift": 383, "intensity": 80, "assignment": "test"}]}},
        }
        profile_b = {
            "formula": "B",
            "commonNames": ["fake B"],
            "spectralSignatures": {"raman": {"peaks": [{"shift": 383, "intensity": 80, "assignment": "test"}]}},
        }
        components = [
            ComponentDeclaration("B", "dopant", 0.1),
            ComponentDeclaration("A", "matrix", 0.9),
        ]
        loader = _loader({"A": profile_a, "B": profile_b})

        result = match_multi_phase(sample, components, loader, "raman")

        # A (matrix) should be first in result.components after sorting
        assert result.components[0].formula == "A"
        # A captures the peak; B gets nothing
        assert result.components[0].intended_peaks_observed == 1
        assert result.components[1].intended_peaks_observed == 0


class TestProfileNotFound:
    def test_missing_profile_flagged(self):
        sample = [{"shift_cm1": 383, "relative_intensity": 80}]
        components = [
            ComponentDeclaration("MoS2", "matrix", 0.5),
            ComponentDeclaration("UnknownXYZ", "dopant", 0.5),
        ]
        loader = _loader({"MoS2": _mos2_profile()})  # UnknownXYZ missing

        result = match_multi_phase(sample, components, loader, "raman")

        assert "UnknownXYZ" in result.intended_but_not_observed


class TestSerialization:
    def test_result_serializes_to_json(self):
        sample = [{"shift_cm1": 383, "relative_intensity": 80}]
        components = [ComponentDeclaration("MoS2", "matrix", 1.0)]
        loader = _loader({"MoS2": _mos2_profile()})

        result = match_multi_phase(sample, components, loader, "raman")
        d = result.to_dict()

        import json
        json.dumps(d)
        assert "components" in d
        assert "intended_phases" in d
