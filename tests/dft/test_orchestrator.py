"""
Tests for dft.orchestrator — pure DAG state machine (no I/O).

@phase R272w-f (DFT P1)
"""
import os

import pytest

from src.dft.orchestrator import (
    COMPLETED,
    FAILED,
    PENDING,
    RUNNING,
    WorkflowState,
    is_relax,
    relaxed_structure_from_out,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _linear():
    return [
        {"id": "relax", "calcType": "vc-relax", "dependsOn": []},
        {"id": "scf", "calcType": "scf", "dependsOn": ["relax"]},
        {"id": "nscf", "calcType": "nscf", "dependsOn": ["scf"]},
    ]


def _branch():
    # relax → scf → {bands, dos}
    return [
        {"id": "relax", "calcType": "vc-relax", "dependsOn": []},
        {"id": "scf", "calcType": "scf", "dependsOn": ["relax"]},
        {"id": "bands", "calcType": "bands", "dependsOn": ["scf"]},
        {"id": "dos", "calcType": "dos", "dependsOn": ["scf"]},
    ]


def test_linear_scheduling():
    w = WorkflowState(_linear())
    assert w.next_runnable() == ["relax"]
    assert w.overall_status() == PENDING
    w.mark_running("relax")
    assert w.overall_status() == RUNNING
    assert w.next_runnable() == []  # scf still blocked
    w.mark_completed("relax")
    assert w.next_runnable() == ["scf"]
    w.mark_completed("scf")
    assert w.next_runnable() == ["nscf"]
    w.mark_completed("nscf")
    assert w.next_runnable() == []
    assert w.is_done() and w.overall_status() == COMPLETED


def test_branch_runs_both_dependents():
    w = WorkflowState(_branch())
    w.mark_completed("relax")
    w.mark_completed("scf")
    assert sorted(w.next_runnable()) == ["bands", "dos"]


def test_failure_propagates_and_stops_branch():
    w = WorkflowState(_linear())
    w.mark_completed("relax")
    newly = w.mark_failed("scf", "SCF did not converge")
    assert sorted(newly) == ["nscf", "scf"]  # nscf blocked by scf
    assert w.states["scf"].status == FAILED
    assert w.states["nscf"].status == FAILED
    assert "upstream" in (w.states["nscf"].error_message or "")
    assert w.is_done() and w.overall_status() == FAILED


def test_branch_failure_only_kills_downstream():
    w = WorkflowState(_branch())
    w.mark_completed("relax")
    w.mark_completed("scf")
    w.mark_failed("bands", "boom")
    # dos is independent of bands → still runnable / pending
    assert w.states["dos"].status == PENDING
    assert w.next_runnable() == ["dos"]
    assert w.overall_status() == FAILED  # a unit failed


def test_unknown_dependency_raises():
    with pytest.raises(ValueError, match="unknown unit"):
        WorkflowState([{"id": "scf", "calcType": "scf", "dependsOn": ["ghost"]}])


def test_cycle_raises():
    with pytest.raises(ValueError, match="cycle"):
        WorkflowState([
            {"id": "a", "calcType": "scf", "dependsOn": ["b"]},
            {"id": "b", "calcType": "scf", "dependsOn": ["a"]},
        ])


def test_duplicate_id_raises():
    with pytest.raises(ValueError, match="duplicate"):
        WorkflowState([
            {"id": "x", "calcType": "scf", "dependsOn": []},
            {"id": "x", "calcType": "nscf", "dependsOn": []},
        ])


def test_snapshot_roundtrip_resumes():
    w = WorkflowState(_linear(), now=lambda: 100.0)
    w.mark_completed("relax")
    w.mark_running("scf")
    snap = w.snapshot()
    assert snap["relax"]["status"] == COMPLETED
    assert snap["scf"]["status"] == RUNNING and snap["scf"]["startedAt"] == 100.0

    w2 = WorkflowState(_linear())
    w2.load_snapshot(snap)
    assert w2.states["relax"].status == COMPLETED
    assert w2.next_runnable() == []  # scf is running, nscf blocked


def test_is_relax():
    assert is_relax("vc-relax") and is_relax("relax")
    assert not is_relax("scf") and not is_relax(None)


def test_relaxed_structure_handoff_from_out():
    out = open(os.path.join(FIXTURES, "synthetic_relax.out")).read()
    base = {"atomicSpecies": [
        {"element": "Ti", "mass": 47.867, "pseudoFile": "Ti.upf"},
        {"element": "O", "mass": 15.999, "pseudoFile": "O.upf"},
    ]}
    s = relaxed_structure_from_out(out, base)
    assert s is not None
    assert s["ibrav"] == 0
    assert s["nat"] == 3 and s["ntyp"] == 2
    assert len(s["cellParameters"]) == 3 and len(s["cellParameters"][0]) == 3
    assert [a["element"] for a in s["atomicPositions"]] == ["Ti", "Ti", "O"]
    assert s["atomicSpecies"] == base["atomicSpecies"]  # carried over
    assert s["positionsType"] == "crystal"


def test_relaxed_structure_none_for_non_relax():
    scf = open(os.path.join(FIXTURES, "synthetic_scf.out")).read()
    assert relaxed_structure_from_out(scf, {}) is None


def test_resume_after_completion_finds_next():
    # resume mid-branch: relax+scf done, bands done, dos pending → dos runnable
    w = WorkflowState(_branch())
    w.mark_completed("relax")
    w.mark_completed("scf")
    w.mark_completed("bands")
    assert w.next_runnable() == ["dos"]
    assert not w.is_done()
    w.mark_completed("dos")
    assert w.is_done() and w.overall_status() == COMPLETED


def test_depends_on_and_ancestors():
    w = WorkflowState(_branch())  # relax → scf → {bands, dos}
    assert w.depends_on("scf") == ("relax",)
    assert w.depends_on("relax") == ()
    assert w.ancestors("bands") == {"scf", "relax"}
    assert w.ancestors("scf") == {"relax"}
    assert w.ancestors("relax") == set()
