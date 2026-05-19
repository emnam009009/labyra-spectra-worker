"""Unit tests for composite physics rules R11-R15.

@phase R185-6-composite-rules
"""
from __future__ import annotations

from src.deviation.composite_rules import run_composite_rules
from src.deviation.multi_phase import ComponentMatch, MultiPhaseResult


def _make_component(
    formula: str,
    role: str = "matrix",
    intent_coverage: float = 1.0,
    matches: list[dict] | None = None,
) -> ComponentMatch:
    return ComponentMatch(
        formula=formula,
        role=role,
        weight_prior=0.9,
        nominal_fraction=0.5,
        reference_label=formula,
        match_result={
            "spectrum_type": "raman",
            "reference_formula": formula,
            "reference_label": formula,
            "tolerance_used": 5.0,
            "matches": matches or [],
            "unmatched_sample": [],
            "unmatched_ref": [],
            "match_count": len(matches or []),
            "match_rate": 1.0,
            "mean_abs_deviation": 0,
            "max_abs_deviation": 0,
            "rmse": 0,
            "quality_grade": "excellent",
        },
        intended_peaks_observed=len(matches or []),
        intended_peaks_total=len(matches or []),
        intent_coverage=intent_coverage,
    )


def _match(s_pos, r_pos, intensity=100, assignment="A1g"):
    return {
        "sample_index": 0, "sample_position": s_pos, "sample_intensity": intensity,
        "sample_fwhm": 8.0, "ref_index": 0, "ref_position": r_pos, "ref_intensity": intensity,
        "ref_assignment": assignment, "deviation": s_pos - r_pos, "confidence": 0.9,
    }


# ── R11 Charge transfer ──────────────────────────────────────────────────────

class TestChargeTransfer:
    def test_mos2_rgo_with_signature_fires(self):
        mos2 = _make_component("MoS2", matches=[
            _match(383, 383, assignment="E12g in-plane"),
            _match(405, 408, intensity=100, assignment="A1g out-of-plane"),  # -3 cm-1
        ])
        carbon = _make_component("C", role="support", matches=[
            _match(1350, 1350, assignment="D-band"),
            _match(1585, 1580, intensity=100, assignment="G-band"),  # +5 cm-1
        ])
        result = MultiPhaseResult(
            spectrum_type="raman",
            components=[mos2, carbon],
            intended_phases=["MoS2", "C"],
        )
        hyps = run_composite_rules(result)
        ct = [h for h in hyps if h.rule_id == "R11-charge-transfer"]
        assert len(ct) == 1
        assert ct[0].citation.doi.startswith("10.1021/nn5025654")

    def test_no_signature_without_both_shifts(self):
        # Both unshifted
        mos2 = _make_component("MoS2", matches=[
            _match(408, 408, assignment="A1g"),
        ])
        carbon = _make_component("C", matches=[
            _match(1580, 1580, assignment="G-band"),
        ])
        result = MultiPhaseResult(
            spectrum_type="raman",
            components=[mos2, carbon],
            intended_phases=["MoS2", "C"],
        )
        hyps = run_composite_rules(result)
        assert not any(h.rule_id == "R11-charge-transfer" for h in hyps)

    def test_single_phase_does_not_fire(self):
        mos2 = _make_component("MoS2", matches=[_match(405, 408, assignment="A1g")])
        result = MultiPhaseResult(
            spectrum_type="raman", components=[mos2], intended_phases=["MoS2"],
        )
        hyps = run_composite_rules(result)
        assert not any(h.rule_id == "R11-charge-transfer" for h in hyps)


# ── R13 Interface phonon ─────────────────────────────────────────────────────

class TestInterfacePhonon:
    def test_low_freq_unmatched_with_two_phases_fires(self):
        mos2 = _make_component("MoS2", matches=[_match(408, 408, assignment="A1g")])
        carbon = _make_component("C", matches=[_match(1580, 1580, assignment="G")])
        result = MultiPhaseResult(
            spectrum_type="raman", components=[mos2, carbon],
            intended_phases=["MoS2", "C"],
            unassigned_peaks=[
                {"sample_index": 0, "position": 35, "intensity": 25, "note": ""},
            ],
        )
        hyps = run_composite_rules(result)
        intf = [h for h in hyps if h.rule_id == "R13-interface-phonon"]
        assert len(intf) == 1

    def test_no_low_freq_no_fire(self):
        mos2 = _make_component("MoS2", matches=[_match(408, 408, assignment="A1g")])
        carbon = _make_component("C", matches=[_match(1580, 1580, assignment="G")])
        result = MultiPhaseResult(
            spectrum_type="raman", components=[mos2, carbon],
            intended_phases=["MoS2", "C"],
        )
        hyps = run_composite_rules(result)
        assert not any(h.rule_id == "R13-interface-phonon" for h in hyps)


# ── R14 Defect-mediated coupling ─────────────────────────────────────────────

class TestDefectMediatedCoupling:
    def test_high_dg_ratio_fires(self):
        mos2 = _make_component("MoS2", matches=[_match(408, 408, assignment="A1g")])
        carbon = _make_component("C", matches=[
            _match(1350, 1350, intensity=180, assignment="D-band"),
            _match(1580, 1580, intensity=100, assignment="G-band"),
        ])
        result = MultiPhaseResult(
            spectrum_type="raman", components=[mos2, carbon],
            intended_phases=["MoS2", "C"],
        )
        hyps = run_composite_rules(result)
        dg = [h for h in hyps if h.rule_id == "R14-defect-mediated-coupling"]
        assert len(dg) == 1
        assert "defective" in dg[0].evidence[0]

    def test_normal_dg_does_not_fire(self):
        mos2 = _make_component("MoS2", matches=[_match(408, 408, assignment="A1g")])
        carbon = _make_component("C", matches=[
            _match(1350, 1350, intensity=100, assignment="D-band"),
            _match(1580, 1580, intensity=100, assignment="G-band"),
        ])
        result = MultiPhaseResult(
            spectrum_type="raman", components=[mos2, carbon],
            intended_phases=["MoS2", "C"],
        )
        hyps = run_composite_rules(result)
        assert not any(h.rule_id == "R14-defect-mediated-coupling" for h in hyps)


# ── R15 vdW stacking ─────────────────────────────────────────────────────────

class TestVdwStacking:
    def test_ultra_low_freq_with_2d_material_fires(self):
        mos2 = _make_component("MoS2", matches=[_match(408, 408, assignment="A1g")])
        result = MultiPhaseResult(
            spectrum_type="raman", components=[mos2],
            intended_phases=["MoS2"],
            unassigned_peaks=[
                {"sample_index": 0, "position": 30, "intensity": 15, "note": ""},
            ],
        )
        hyps = run_composite_rules(result)
        vdw = [h for h in hyps if h.rule_id == "R15-vdw-stacking-modes"]
        assert len(vdw) == 1


# ── Engine ordering ──────────────────────────────────────────────────────────

class TestEngineOrdering:
    def test_hypotheses_sorted_by_confidence(self):
        mos2 = _make_component("MoS2", matches=[_match(405, 408, assignment="A1g")])
        carbon = _make_component("C", role="support", matches=[
            _match(1350, 1350, intensity=180, assignment="D-band"),
            _match(1585, 1580, intensity=100, assignment="G-band"),
        ])
        result = MultiPhaseResult(
            spectrum_type="raman", components=[mos2, carbon],
            intended_phases=["MoS2", "C"],
        )
        hyps = run_composite_rules(result)
        for i in range(len(hyps) - 1):
            assert hyps[i].confidence >= hyps[i + 1].confidence
