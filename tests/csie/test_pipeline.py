"""Integration tests for CSIE pipeline.

Mocks Firestore I/O. Real Firestore tested manually post-deploy.

@phase R185-8b-csie-integration
"""
from __future__ import annotations

import pytest
from unittest.mock import patch

from src.csie.pipeline import run_csie_for_sample


def _comp(formula: str, role: str = "matrix"):
    return {"formula": formula, "role": role, "nominalFraction": 0.5}


def _measurement(spectrum_id: str, sp_type: str, formulas: list[tuple[str, str, float]],
                 analyzed_at: int = 1_000_000):
    components = [
        {"formula": f, "intent_coverage": cov, "match_result": {"quality_grade": q}}
        for f, q, cov in formulas
    ]
    return {
        "spectrumId": spectrum_id,
        "spectrumType": sp_type,
        "analyzedAt": analyzed_at,
        "analysisResult": {
            "deviationAnalysis": {
                "mode": "multi-phase",
                "multiPhase": {"components": components, "unassigned_peaks": []},
                "perComponentHypotheses": {f: [] for f, _, _ in formulas},
            },
        },
    }


class TestPipelineHappyPath:
    def test_runs_when_sample_has_composition_and_measurements(self):
        with patch("src.csie.pipeline.check_rate_limit", return_value=True), \
             patch("src.csie.pipeline.fetch_sample", return_value={
                 "id": "sample-001", "tenantId": "tenant-dev-001",
                 "composition": [_comp("MoS2"), _comp("C", "support")],
             }), \
             patch("src.csie.pipeline.fetch_analyzed_measurements", return_value=[
                 _measurement("s1", "raman", [("MoS2", "excellent", 1.0), ("C", "good", 0.9)]),
                 _measurement("s2", "xrd",   [("MoS2", "excellent", 1.0), ("C", "good", 0.85)]),
             ]), \
             patch("src.csie.pipeline.should_skip_debounce", return_value=False), \
             patch("src.csie.pipeline.write_csie_result"):

            result = run_csie_for_sample("tenant-dev-001", "sample-001")
            assert result.status == "ok"
            assert result.consistency is not None
            assert len(result.consistency.declared_phases) == 2


class TestPipelineGuards:
    def test_rate_limited(self):
        with patch("src.csie.pipeline.check_rate_limit", return_value=False):
            result = run_csie_for_sample("tenant-dev-001", "sample-001")
            assert result.status == "rate_limited"

    def test_sample_not_found(self):
        with patch("src.csie.pipeline.check_rate_limit", return_value=True), \
             patch("src.csie.pipeline.fetch_sample", return_value=None):
            result = run_csie_for_sample("tenant-dev-001", "sample-001")
            assert result.status == "failed"
            assert any("not found" in n.lower() or "access" in n.lower() for n in result.notes)

    def test_no_composition(self):
        with patch("src.csie.pipeline.check_rate_limit", return_value=True), \
             patch("src.csie.pipeline.fetch_sample", return_value={
                 "id": "s1", "tenantId": "tenant-dev-001", "composition": [],
             }):
            result = run_csie_for_sample("tenant-dev-001", "sample-001")
            assert result.status == "insufficient_data"

    def test_single_measurement(self):
        with patch("src.csie.pipeline.check_rate_limit", return_value=True), \
             patch("src.csie.pipeline.fetch_sample", return_value={
                 "id": "s1", "tenantId": "tenant-dev-001",
                 "composition": [_comp("MoS2")],
             }), \
             patch("src.csie.pipeline.fetch_analyzed_measurements", return_value=[
                 _measurement("only", "raman", [("MoS2", "good", 0.8)]),
             ]):
            result = run_csie_for_sample("tenant-dev-001", "sample-001")
            assert result.status == "insufficient_data"


class TestPipelineDebounce:
    def test_debounce_skips_recomputation(self):
        with patch("src.csie.pipeline.check_rate_limit", return_value=True), \
             patch("src.csie.pipeline.fetch_sample", return_value={
                 "id": "s1", "tenantId": "tenant-dev-001",
                 "composition": [_comp("MoS2")],
             }), \
             patch("src.csie.pipeline.fetch_analyzed_measurements", return_value=[
                 _measurement("s1", "raman", [("MoS2", "good", 0.8)]),
                 _measurement("s2", "xrd",   [("MoS2", "good", 0.8)]),
             ]), \
             patch("src.csie.pipeline.should_skip_debounce", return_value=True), \
             patch("src.csie.pipeline.write_csie_result") as mock_write:
            result = run_csie_for_sample("tenant-dev-001", "sample-001")
            assert result.status == "ok"
            # Write should NOT happen when debounce triggers
            mock_write.assert_not_called()

    def test_force_bypasses_debounce(self):
        with patch("src.csie.pipeline.check_rate_limit", return_value=True), \
             patch("src.csie.pipeline.fetch_sample", return_value={
                 "id": "s1", "tenantId": "tenant-dev-001",
                 "composition": [_comp("MoS2"), _comp("C", "support")],
             }), \
             patch("src.csie.pipeline.fetch_analyzed_measurements", return_value=[
                 _measurement("s1", "raman", [("MoS2", "good", 0.8), ("C", "good", 0.8)]),
                 _measurement("s2", "xrd",   [("MoS2", "good", 0.8), ("C", "good", 0.8)]),
             ]), \
             patch("src.csie.pipeline.should_skip_debounce", return_value=True), \
             patch("src.csie.pipeline.write_csie_result") as mock_write:
            result = run_csie_for_sample("tenant-dev-001", "sample-001", force=True)
            assert result.status == "ok"
            mock_write.assert_called_once()
