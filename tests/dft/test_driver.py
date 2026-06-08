"""
Tests for dft.driver.advance — event-driven DAG tick with a fake DftIO (no GCP).

@phase R272w-h (DFT P1-3)
"""
import os

from src.dft.driver import advance

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _units():
    # relax → scf → {bands, dos}
    return [
        {"id": "relax", "calcType": "vc-relax", "dependsOn": []},
        {"id": "scf", "calcType": "scf", "dependsOn": ["relax"]},
        {"id": "bands", "calcType": "bands", "dependsOn": ["scf"]},
        {"id": "dos", "calcType": "dos", "dependsOn": ["scf"]},
    ]


class FakeIO:
    """In-memory DftIO: records launches, serves canned .out, persists snapshot."""

    def __init__(self, units, structure, out_map=None):
        self._doc = {
            "units": units,
            "structure": structure,
            "global": {"ecutwfc": 50},
            "snapshot": {},
            "relaxedStructures": {},
        }
        self.launched = []           # (unit_id, calc, gcs_deps) in order
        self.launch_structures = {}  # unit_id -> structure passed
        self._out_map = out_map or {}
        self.saves = 0

    def load(self, t, w):
        return {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
                for k, v in self._doc.items()}

    def launch(self, t, w, uid, calc, structure, gp, deps):
        self.launched.append((uid, calc, list(deps)))
        self.launch_structures[uid] = structure
        return f"job-{uid}"

    def fetch_output(self, t, w, uid):
        return self._out_map[uid]

    def save(self, t, w, snapshot, overall, relaxed, results=None):
        self.saves += 1
        self._doc["snapshot"] = snapshot
        self._doc["relaxedStructures"] = relaxed
        self.results = results


BASE = {"ibrav": 4, "atomicSpecies": [{"element": "W"}, {"element": "O"}]}


def test_start_launches_only_root():
    io = FakeIO(_units(), BASE)
    overall = advance(io, "t1", "w1", event=None)
    assert [u for (u, _c, _d) in io.launched] == ["relax"]
    assert overall == "running"
    assert io._doc["snapshot"]["relax"]["status"] == "queued"


def test_relax_success_handoff_then_launch_scf():
    relax_out = open(os.path.join(FIXTURES, "synthetic_relax.out")).read()
    io = FakeIO(_units(), BASE, out_map={"relax": relax_out})
    advance(io, "t1", "w1", event=None)               # → launch relax
    io.launched.clear()
    overall = advance(io, "t1", "w1", {"unitId": "relax", "state": "SUCCEEDED"})
    # scf launched, with relaxed structure (ibrav=0 from handoff) + NO outdir deps
    assert [u for (u, _c, _d) in io.launched] == ["scf"]
    assert io.launched[0][2] == []                    # relax dep is structure-handoff, not outdir
    assert io.launch_structures["scf"]["ibrav"] == 4  # relaxed structure preserves base hexagonal ibrav=4
    assert "celldm" in io.launch_structures["scf"]    # celldm recomputed from relaxed cell
    assert "relax" in io._doc["relaxedStructures"]
    assert overall == "running"


def test_scf_success_launches_branch_with_outdir_dep():
    relax_out = open(os.path.join(FIXTURES, "synthetic_relax.out")).read()
    io = FakeIO(_units(), BASE, out_map={"relax": relax_out})
    advance(io, "t1", "w1", event=None)
    advance(io, "t1", "w1", {"unitId": "relax", "state": "SUCCEEDED"})  # → scf
    io.launched.clear()
    advance(io, "t1", "w1", {"unitId": "scf", "state": "SUCCEEDED"})    # → bands + dos
    launched = sorted((u, tuple(d)) for (u, _c, d) in io.launched)
    assert launched == [("bands", ("scf",)), ("dos", ("scf",))]        # restart from scf outdir
    # both inherit the relaxed structure (relax is transitive ancestor)
    assert io.launch_structures["bands"]["ibrav"] == 4
    assert io.launch_structures["dos"]["ibrav"] == 4


def test_relax_failure_propagates_no_launch():
    io = FakeIO(_units(), BASE)
    advance(io, "t1", "w1", event=None)
    io.launched.clear()
    overall = advance(io, "t1", "w1", {"unitId": "relax", "state": "FAILED"})
    assert io.launched == []                       # whole DAG blocked
    assert overall == "failed"
    assert io._doc["snapshot"]["scf"]["status"] == "failed"
    assert io._doc["snapshot"]["dos"]["status"] == "failed"


def test_running_event_is_noop():
    io = FakeIO(_units(), BASE)
    advance(io, "t1", "w1", event=None)
    io.launched.clear()
    overall = advance(io, "t1", "w1", {"unitId": "relax", "state": "RUNNING"})
    assert io.launched == []      # nothing newly runnable
    assert overall == "running"   # relax still queued/running


def test_full_run_to_completed():
    relax_out = open(os.path.join(FIXTURES, "synthetic_relax.out")).read()
    io = FakeIO(_units(), BASE, out_map={"relax": relax_out})
    advance(io, "t1", "w1", event=None)
    advance(io, "t1", "w1", {"unitId": "relax", "state": "SUCCEEDED"})
    advance(io, "t1", "w1", {"unitId": "scf", "state": "SUCCEEDED"})
    advance(io, "t1", "w1", {"unitId": "bands", "state": "SUCCEEDED"})
    overall = advance(io, "t1", "w1", {"unitId": "dos", "state": "SUCCEEDED"})
    assert overall == "completed"
