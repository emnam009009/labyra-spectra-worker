"""Unit tests for CSIE evidence aggregator + consistency checker.

@phase R185-8a-csie-evidence-aggregation
"""
from __future__ import annotations

import pytest

from src.csie.aggregator import (
    MAX_MEASUREMENTS_PER_RUN,
    aggregate_evidence,
    run_csie,
)
from src.csie.types import CSIEResult


def _comp(formula: str, role: str = "matrix", fraction: float = 0.5) -> dict:
    return {"formula": formula, "role": role, "nominalFraction": fraction}


def _measurement_multi_phase(
    spectrum_id: str,
    spectrum_type: str,
    formulas_with_quality: list[tuple[str, str, float]],  # (formula, quality, coverage)
    unassigned_count: int = 0,
    analyzed_at: int = 1_000_000,
) -> dict:
    components = []
    per_comp_hyps = {}
    for formula, quality, coverage in formulas_with_quality:
        components.append({
            "formula": formula,
            "intent_coverage": coverage,
            "match_result": {"quality_grade": quality},
        })
        per_comp_hyps[formula] = []
    return {
        "spectrumId": spectrum_id,
        "spectrumType": spectrum_type,
        "analyzedAt": analyzed_at,
        "analysisResult": {
            "deviationAnalysis": {
                "mode": "multi-phase",
                "multiPhase": {
                    "components": components,
                    "unassigned_peaks": [
                        {"position": 100 + i, "intensity": 50}
                        for i in range(unassigned_count)
                    ],
                },
                "perComponentHypotheses": per_comp_hyps,
            },
        },
    }


# ── Input validation ─────────────────────────────────────────────────────────

class TestInputValidation:
    def test_invalid_tenant_rejected(self):
        result = run_csie(
            tenant_id="../etc/passwd",
            sample_id="sample-001",
            declared_composition=[_comp("MoS2")],
            measurements=[],
        )
        assert result.status == "failed"
        assert "invalid_tenant_id" in result.notes

    def test_invalid_sample_id_rejected(self):
        result = run_csie(
            tenant_id="tenant-dev-001",
            sample_id="x" * 200,
            declared_composition=[_comp("MoS2")],
            measurements=[],
        )
        assert result.status == "failed"

    def test_empty_tenant_rejected(self):
        result = run_csie("", "s1", [_comp("MoS2")], [])
        assert result.status == "failed"


# ── Insufficient data ────────────────────────────────────────────────────────

class TestInsufficientData:
    def test_single_measurement_insufficient(self):
        result = run_csie(
            "tenant-dev-001", "sample-001",
            [_comp("MoS2")],
            [_measurement_multi_phase("s1", "raman", [("MoS2", "excellent", 1.0)])],
        )
        assert result.status == "insufficient_data"

    def test_no_composition_insufficient(self):
        result = run_csie(
            "tenant-dev-001", "sample-001",
            [],  # empty composition
            [_measurement_multi_phase("s1", "raman", [("MoS2", "excellent", 1.0)])] * 2,
        )
        assert result.status == "insufficient_data"


# ── DoS protection ───────────────────────────────────────────────────────────

class TestDoSProtection:
    def test_truncates_at_max_measurements(self):
        many = [
            _measurement_multi_phase(
                f"s{i}", "raman", [("MoS2", "good", 0.8)], analyzed_at=1000 + i,
            )
            for i in range(50)
        ]
        result = run_csie(
            "tenant-dev-001", "sample-001",
            [_comp("MoS2")],
            many,
        )
        assert result.status == "ok"
        assert result.consistency.measurements_analyzed == MAX_MEASUREMENTS_PER_RUN


# ── Aggregation ──────────────────────────────────────────────────────────────

class TestEvidenceAggregation:
    def test_confirmed_two_techniques(self):
        result = run_csie(
            "tenant-dev-001", "sample-001",
            [_comp("MoS2", "matrix", 0.7), _comp("C", "support", 0.3)],
            [
                _measurement_multi_phase(
                    "s1", "raman",
                    [("MoS2", "excellent", 1.0), ("C", "good", 0.9)],
                ),
                _measurement_multi_phase(
                    "s2", "xrd",
                    [("MoS2", "excellent", 1.0), ("C", "good", 0.85)],
                ),
            ],
        )
        assert result.status == "ok"
        mos2 = [p for p in result.consistency.declared_phases if p.formula == "MoS2"][0]
        assert mos2.verdict == "confirmed"
        assert mos2.consistency_score > 0.8

    def test_partial_when_only_one_technique(self):
        result = run_csie(
            "tenant-dev-001", "sample-001",
            [_comp("MoS2", "matrix"), _comp("WO3", "filler")],
            [
                _measurement_multi_phase(
                    "s1", "raman", [("MoS2", "excellent", 1.0), ("WO3", "poor", 0.0)],
                ),
                _measurement_multi_phase(
                    "s2", "xrd", [("MoS2", "excellent", 1.0), ("WO3", "poor", 0.0)],
                ),
            ],
        )
        wo3 = [p for p in result.consistency.declared_phases if p.formula == "WO3"][0]
        # WO3 missing from both spectra → conflict
        assert wo3.verdict in ("missing", "conflict")
        assert wo3.consistency_score < 0.4

    def test_unexpected_observations_flagged(self):
        result = run_csie(
            "tenant-dev-001", "sample-001",
            [_comp("MoS2")],
            [
                _measurement_multi_phase("s1", "raman", [("MoS2", "good", 0.8)], unassigned_count=5),
                _measurement_multi_phase("s2", "xrd", [("MoS2", "good", 0.8)], unassigned_count=4),
            ],
        )
        assert result.status == "ok"
        assert len(result.consistency.unexpected_observations) >= 1


# ── PII safety ───────────────────────────────────────────────────────────────

class TestPIISafety:
    def test_sample_id_not_in_output(self):
        result = run_csie(
            "tenant-dev-001", "very-sensitive-sample-name-12345",
            [_comp("MoS2")],
            [_measurement_multi_phase(f"s{i}", "raman", [("MoS2", "good", 0.8)])
             for i in range(2)],
        )
        d = result.to_dict()
        import json
        serialized = json.dumps(d)
        # Sample ID must be hashed, not appear in result
        assert "very-sensitive-sample-name-12345" not in serialized


# ── Idempotency ──────────────────────────────────────────────────────────────

class TestIdempotency:
    def test_same_input_same_key(self):
        ms = [_measurement_multi_phase(f"s{i}", "raman",
              [("MoS2", "good", 0.8)], analyzed_at=2000 + i) for i in range(2)]
        comp = [_comp("MoS2")]
        r1 = run_csie("tenant-dev-001", "sample-001", comp, ms)
        r2 = run_csie("tenant-dev-001", "sample-001", comp, ms)
        assert r1.idempotency_key == r2.idempotency_key

    def test_different_max_ts_different_key(self):
        comp = [_comp("MoS2")]
        ms1 = [_measurement_multi_phase(f"s{i}", "raman",
               [("MoS2", "good", 0.8)], analyzed_at=2000 + i) for i in range(2)]
        ms2 = [_measurement_multi_phase(f"s{i}", "raman",
               [("MoS2", "good", 0.8)], analyzed_at=3000 + i) for i in range(2)]
        r1 = run_csie("tenant-dev-001", "sample-001", comp, ms1)
        r2 = run_csie("tenant-dev-001", "sample-001", comp, ms2)
        assert r1.idempotency_key != r2.idempotency_key


# ── Formula validation ───────────────────────────────────────────────────────

class TestFormulaValidation:
    def test_lowercase_first_letter_rejected(self):
        result = run_csie(
            "tenant-dev-001", "sample-001",
            [_comp("mos2")],  # lowercase
            [_measurement_multi_phase(f"s{i}", "raman", [], 0) for i in range(2)],
        )
        # Should not crash; "mos2" not in evidence_map
        assert result.status == "ok"
        assert all(p.formula != "mos2" for p in result.consistency.declared_phases)


# ── Serialization ────────────────────────────────────────────────────────────

class TestSerialization:
    def test_full_result_json_safe(self):
        import json
        result = run_csie(
            "tenant-dev-001", "sample-001",
            [_comp("MoS2")],
            [_measurement_multi_phase(f"s{i}", "raman",
             [("MoS2", "good", 0.8)]) for i in range(2)],
        )
        d = result.to_dict()
        json.dumps(d)  # must serialize cleanly
        assert d["status"] == "ok"
        assert d["consistency"]["measurements_analyzed"] == 2
