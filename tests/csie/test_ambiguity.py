"""Unit tests for ambiguous hypothesis handler.

@phase R185-9-ambiguous-hypothesis-handler
"""
from __future__ import annotations

import pytest

from src.csie.ambiguity import (
    DISCRIMINATION_EXPERIMENTS,
    OBSERVATION_CLUSTERS,
    cluster_hypotheses,
    handle_ambiguous,
    rescore_with_multi_spectrum,
)


def _hyp(rule_id: str, name: str = "", confidence: float = 0.7, citation_doi: str | None = None):
    return {
        "rule_id": rule_id,
        "name": name or rule_id,
        "confidence": confidence,
        "evidence": [f"evidence for {rule_id}"],
        "citation": {"doi": citation_doi} if citation_doi else None,
    }


class TestClustering:
    def test_strain_pcm_charge_transfer_cluster(self):
        hyps = [
            _hyp("R1-tensile-strain", confidence=0.7),
            _hyp("R3-phonon-confinement", confidence=0.65),
            _hyp("R11-charge-transfer", confidence=0.6),
        ]
        clusters = cluster_hypotheses(hyps)
        assert "raman_peak_shift_with_broadening" in clusters
        assert len(clusters["raman_peak_shift_with_broadening"]) == 3

    def test_single_hypothesis_not_clustered(self):
        # Only 1 rule fires → no ambiguity
        hyps = [_hyp("R3-phonon-confinement", confidence=0.8)]
        clusters = cluster_hypotheses(hyps)
        # Must have 2+ to be ambiguous
        assert all(len(v) >= 2 for v in clusters.values())

    def test_interface_phonon_vs_vdw(self):
        hyps = [
            _hyp("R13-interface-phonon", confidence=0.7),
            _hyp("R15-vdw-stacking-modes", confidence=0.65),
        ]
        clusters = cluster_hypotheses(hyps)
        assert "low_freq_unassigned_peak" in clusters

    def test_carbon_disorder_cluster(self):
        hyps = [
            _hyp("R8-amorphization", confidence=0.7),
            _hyp("R14-defect-mediated-coupling", confidence=0.65),
        ]
        clusters = cluster_hypotheses(hyps)
        assert "carbon_disorder_signature" in clusters


class TestHandleAmbiguous:
    def test_empty_input_no_output(self):
        assert handle_ambiguous([]) == []

    def test_strain_pcm_cluster_produces_output(self):
        hyps = [
            _hyp("R1-tensile-strain", confidence=0.7, citation_doi="10.1016/j.solidstatesciences.2014.04.012"),
            _hyp("R3-phonon-confinement", confidence=0.65, citation_doi="10.1103/PhysRevB.63.125415"),
        ]
        ambs = handle_ambiguous(hyps)
        assert len(ambs) == 1
        amb = ambs[0]
        assert amb.observation_id == "raman_peak_shift_with_broadening"
        assert len(amb.candidates) == 2
        # Discrimination experiments populated
        assert len(amb.discrimination_experiments) > 0
        # Citations preserved
        assert amb.candidates[0].citation_doi is not None

    def test_close_scores_yield_warning_severity(self):
        hyps = [
            _hyp("R1-tensile-strain", confidence=0.70),
            _hyp("R3-phonon-confinement", confidence=0.68),
        ]
        ambs = handle_ambiguous(hyps)
        assert len(ambs) == 1
        # Within 0.15 → warning at minimum
        assert ambs[0].severity in ("warning", "error")

    def test_very_close_scores_yield_error_severity(self):
        hyps = [
            _hyp("R1-tensile-strain", confidence=0.65),
            _hyp("R3-phonon-confinement", confidence=0.64),
        ]
        ambs = handle_ambiguous(hyps)
        assert len(ambs) == 1
        # Within 0.05 → error
        assert ambs[0].severity == "error"

    def test_discrimination_experiments_attached(self):
        hyps = [
            _hyp("R1-tensile-strain", confidence=0.7),
            _hyp("R3-phonon-confinement", confidence=0.65),
        ]
        ambs = handle_ambiguous(hyps)
        assert len(ambs[0].discrimination_experiments) >= 2
        # TEM should be one of them for size discrimination
        techniques = {e.technique for e in ambs[0].discrimination_experiments}
        assert any("TEM" in t for t in techniques)

    def test_multi_cluster_sorted_by_severity(self):
        hyps = [
            # Cluster 1: very close (error)
            _hyp("R1-tensile-strain", confidence=0.65),
            _hyp("R3-phonon-confinement", confidence=0.64),
            # Cluster 2: clear winner (info)
            _hyp("R13-interface-phonon", confidence=0.85),
            _hyp("R15-vdw-stacking-modes", confidence=0.4),
        ]
        ambs = handle_ambiguous(hyps)
        # Error severity should come first
        assert ambs[0].severity == "error"


class TestRescoring:
    def test_no_consistency_returns_base(self):
        hyp = _hyp("R3-phonon-confinement", confidence=0.6)
        rescored = rescore_with_multi_spectrum(hyp, None)
        assert rescored == 0.6

    def test_confirmed_phase_boosts_score(self):
        hyp = _hyp("R3-phonon-confinement", confidence=0.6)
        consistency = {
            "declared_phases": [
                {"verdict": "confirmed", "consistency_score": 0.9},
            ],
        }
        rescored = rescore_with_multi_spectrum(hyp, consistency)
        assert rescored > 0.6

    def test_boost_capped_at_095(self):
        hyp = _hyp("R3-phonon-confinement", confidence=0.9)
        consistency = {
            "declared_phases": [
                {"verdict": "confirmed", "consistency_score": 1.0},
            ],
        }
        rescored = rescore_with_multi_spectrum(hyp, consistency)
        assert rescored <= 0.95


class TestDiscriminationKnowledgeBase:
    def test_all_clusters_have_discrimination(self):
        for cluster_id, cluster_def in OBSERVATION_CLUSTERS.items():
            disc_ids = cluster_def.get("discrimination", [])
            assert len(disc_ids) > 0, f"Cluster {cluster_id} has no discrimination experiments"
            # All referenced exp IDs must exist
            for disc_id in disc_ids:
                assert disc_id in DISCRIMINATION_EXPERIMENTS, (
                    f"Cluster {cluster_id} references unknown exp {disc_id}"
                )

    def test_all_experiments_have_outcomes(self):
        for exp_id, exp in DISCRIMINATION_EXPERIMENTS.items():
            assert exp.technique, f"{exp_id} missing technique"
            assert exp.measurement, f"{exp_id} missing measurement protocol"
            assert exp.expected_outcomes, f"{exp_id} missing expected outcomes"


class TestSerialization:
    def test_ambiguous_observation_json_safe(self):
        import json
        hyps = [
            _hyp("R1-tensile-strain", confidence=0.7, citation_doi="10.1016/foo"),
            _hyp("R3-phonon-confinement", confidence=0.68),
        ]
        ambs = handle_ambiguous(hyps)
        # Must serialize cleanly
        json.dumps([a.to_dict() for a in ambs])
