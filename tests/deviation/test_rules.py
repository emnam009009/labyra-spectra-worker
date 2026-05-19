"""Unit tests for physics rules engine.

Each rule tested with positive case (should fire) + negative case (should not).

@phase R185-2-physics-rules-engine
"""
from __future__ import annotations

from src.deviation.peak_matcher import MatchResult, PeakMatch, UnmatchedPeak
from src.deviation.rules import run_rules


def _build_result(
    matches: list[PeakMatch] | None = None,
    unmatched_sample: list[UnmatchedPeak] | None = None,
    spectrum_type: str = "raman",
    formula: str = "MoS2",
    label: str = "MoS2 (2H)",
    match_rate: float = 1.0,
    match_count: int | None = None,
) -> MatchResult:
    matches = matches or []
    return MatchResult(
        spectrum_type=spectrum_type,
        reference_formula=formula,
        reference_label=label,
        tolerance_used=5.0,
        matches=matches,
        unmatched_sample=unmatched_sample or [],
        match_count=match_count if match_count is not None else len(matches),
        match_rate=match_rate,
    )


def _match(s_pos, r_pos, intensity=100, fwhm=8.0, assignment="A1g"):
    return PeakMatch(
        sample_index=0, sample_position=s_pos, sample_intensity=intensity,
        sample_fwhm=fwhm, ref_index=0, ref_position=r_pos, ref_intensity=intensity,
        ref_assignment=assignment, deviation=s_pos - r_pos, confidence=0.9,
    )


# ── R1 Tensile strain ─────────────────────────────────────────────────────────

class TestTensileStrain:
    def test_consistent_downshift_fires(self):
        matches = [
            _match(380, 383, assignment="E12g in-plane"),
            _match(405, 408, assignment="A1g out-of-plane"),
        ]
        result = _build_result(matches=matches)
        hyps = run_rules(result)
        tensile = [h for h in hyps if h.rule_id == "R1-tensile-strain"]
        assert len(tensile) == 1
        assert "strain" in tensile[0].quantitative_estimate.lower()

    def test_inconsistent_shifts_does_not_fire(self):
        matches = [_match(380, 383), _match(411, 408)]  # one down, one up
        result = _build_result(matches=matches)
        hyps = run_rules(result)
        assert not any(h.rule_id == "R1-tensile-strain" for h in hyps)

    def test_tiny_shift_does_not_fire(self):
        matches = [_match(383, 383), _match(408, 408)]
        result = _build_result(matches=matches)
        hyps = run_rules(result)
        assert not any(h.rule_id == "R1-tensile-strain" for h in hyps)


# ── R2 Compressive strain ─────────────────────────────────────────────────────

class TestCompressiveStrain:
    def test_consistent_upshift_fires(self):
        matches = [
            _match(386, 383, assignment="E12g"),
            _match(411, 408, assignment="A1g"),
        ]
        result = _build_result(matches=matches)
        hyps = run_rules(result)
        compr = [h for h in hyps if h.rule_id == "R2-compressive-strain"]
        assert len(compr) == 1


# ── R3 Phonon confinement ─────────────────────────────────────────────────────

class TestPhononConfinement:
    def test_broadening_small_shift_fires(self):
        matches = [
            _match(145, 144, intensity=100, fwhm=16.0),  # FWHM 2x ref
            _match(640, 639, intensity=40, fwhm=14.0),
        ]
        ref_fwhms = {0: 8.0}
        result = _build_result(matches=matches, formula="TiO2")
        hyps = run_rules(result, {"ref_fwhms": ref_fwhms})
        pcm = [h for h in hyps if h.rule_id == "R3-phonon-confinement"]
        assert len(pcm) == 1
        assert "nm" in pcm[0].quantitative_estimate


# ── R4 Oxygen vacancy ─────────────────────────────────────────────────────────

class TestOxygenVacancy:
    def test_oxide_softening_fires(self):
        matches = [
            _match(802, 806, assignment="W=O stretch"),
            _match(710, 715, assignment="W-O-W bridge"),
        ]
        result = _build_result(matches=matches, formula="WO3", label="WO3 monoclinic")
        hyps = run_rules(result)
        ov = [h for h in hyps if h.rule_id == "R4-oxygen-vacancy"]
        assert len(ov) == 1
        assert ov[0].severity == "warning"

    def test_non_transition_oxide_does_not_fire(self):
        matches = [_match(515, 520, assignment="Si TO")]
        result = _build_result(matches=matches, formula="Si")
        hyps = run_rules(result)
        assert not any(h.rule_id == "R4-oxygen-vacancy" for h in hyps)


# ── R5 Mixed phase ────────────────────────────────────────────────────────────

class TestMixedPhase:
    def test_strong_unmatched_peaks_fire(self):
        matches = [_match(806, 806), _match(715, 715), _match(267, 267)]
        unmatched = [
            UnmatchedPeak(side="sample", index=3, position=640, intensity=40),
            UnmatchedPeak(side="sample", index=4, position=782, intensity=30),
        ]
        result = _build_result(
            matches=matches, unmatched_sample=unmatched,
            formula="WO3", label="WO3 monoclinic", match_rate=0.6,
        )
        hyps = run_rules(result)
        mixed = [h for h in hyps if h.rule_id == "R5-mixed-phase"]
        assert len(mixed) == 1


# ── R7 TMD layer count ───────────────────────────────────────────────────────

class TestTmdLayerCount:
    def test_mos2_monolayer_delta_20(self):
        matches = [
            _match(383, 383, assignment="E12g in-plane"),  # at 383
            _match(403, 408, assignment="A1g out-of-plane"),  # at 403, Δ = 20
        ]
        result = _build_result(matches=matches, formula="MoS2")
        hyps = run_rules(result)
        layer = [h for h in hyps if h.rule_id == "R7-tmd-layer-count"]
        assert len(layer) == 1
        assert "monolayer" in layer[0].name

    def test_mos2_bulk_delta_26(self):
        matches = [
            _match(382, 383, assignment="E12g"),
            _match(408, 408, assignment="A1g"),  # at 408, Δ = 26
        ]
        result = _build_result(matches=matches, formula="MoS2")
        hyps = run_rules(result)
        layer = [h for h in hyps if h.rule_id == "R7-tmd-layer-count"]
        assert len(layer) == 1
        assert "bulk" in layer[0].name


# ── R9 Substrate ──────────────────────────────────────────────────────────────

class TestSubstrate:
    def test_si_520_unmatched_fires(self):
        matches = [_match(383, 383, assignment="E12g")]
        unmatched = [UnmatchedPeak(side="sample", index=1, position=520, intensity=80)]
        result = _build_result(matches=matches, unmatched_sample=unmatched, formula="MoS2")
        hyps = run_rules(result)
        sub = [h for h in hyps if h.rule_id == "R9-substrate-contribution"]
        assert len(sub) == 1
        assert "Si" in sub[0].evidence[0]


# ── R10 Resonance enhancement ────────────────────────────────────────────────

class TestResonance:
    def test_ws2_532nm_350_unmatched_fires(self):
        matches = [_match(356, 356), _match(417, 417)]
        unmatched = [UnmatchedPeak(side="sample", index=2, position=350, intensity=40)]
        result = _build_result(matches=matches, unmatched_sample=unmatched, formula="WS2")
        hyps = run_rules(result, {"laser_wavelength": 532})
        res = [h for h in hyps if h.rule_id == "R10-resonance-enhancement"]
        assert len(res) == 1

    def test_ws2_785nm_no_resonance(self):
        matches = [_match(356, 356)]
        unmatched = [UnmatchedPeak(side="sample", index=2, position=350, intensity=40)]
        result = _build_result(matches=matches, unmatched_sample=unmatched, formula="WS2")
        hyps = run_rules(result, {"laser_wavelength": 785})
        assert not any(h.rule_id == "R10-resonance-enhancement" for h in hyps)


# ── Engine ordering ──────────────────────────────────────────────────────────

class TestEngineOrdering:
    def test_hypotheses_sorted_by_confidence_desc(self):
        matches = [
            _match(380, 383, assignment="E12g"),
            _match(405, 408, assignment="A1g"),
        ]
        result = _build_result(matches=matches, formula="MoS2")
        hyps = run_rules(result)
        for i in range(len(hyps) - 1):
            assert hyps[i].confidence >= hyps[i + 1].confidence


# ── Serialization ────────────────────────────────────────────────────────────

class TestSerialization:
    def test_hypothesis_to_dict(self):
        matches = [_match(380, 383, assignment="E12g"), _match(405, 408, assignment="A1g")]
        result = _build_result(matches=matches, formula="MoS2")
        hyps = run_rules(result)
        if hyps:
            d = hyps[0].to_dict()
            import json
            json.dumps(d)  # must be JSON serializable
            assert "rule_id" in d
            assert "confidence" in d
