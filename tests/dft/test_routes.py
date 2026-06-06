"""TestClient tests for the DFT router, isolated from main.py / heavy worker deps.

@phase R272w-c (DFT P0 — endpoints)
"""
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pymatgen.core import Lattice, Structure

from src.dft.routes import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def _rutile_cif():
    return Structure.from_spacegroup(
        136, Lattice.tetragonal(4.594, 2.959), ["Ti", "O"], [[0, 0, 0], [0.305, 0.305, 0]]
    ).to(fmt="cif")


def test_structure_from_cif():
    r = client.post("/dft/structure", json={
        "source": "cif", "cif_text": _rutile_cif(), "pseudo_map": {"Ti": "Ti.UPF", "O": "O.UPF"},
    })
    assert r.status_code == 200
    s = r.json()
    assert s["ibrav"] == 6 and s["ntyp"] == 2 and s["nat"] == 6
    assert all(d["pseudoFile"] for d in s["atomicSpecies"])


def test_structure_bad_cif_400():
    assert client.post("/dft/structure", json={"source": "cif", "cif_text": "nope"}).status_code == 400


def test_structure_missing_field_400():
    assert client.post("/dft/structure", json={"source": "cif"}).status_code == 400


def test_kpath():
    r = client.post("/dft/kpath", json={"source": "cif", "cif_text": _rutile_cif()})
    assert r.status_code == 200
    body = r.json()
    assert body["path"] and body["path"][-1]["npoints"] == 1


def test_generate_full_dag():
    cif = _rutile_cif()
    s = client.post("/dft/structure", json={
        "source": "cif", "cif_text": cif, "pseudo_map": {"Ti": "Ti.UPF", "O": "O.UPF"},
    }).json()
    path = client.post("/dft/kpath", json={"source": "cif", "cif_text": cif}).json()["path"]
    r = client.post("/dft/generate", json={
        "structure": s, "prefix": "TiO2", "functional": "pbe", "ecutwfc": 50, "ecutrho": 400,
        "hubbard": [{"manifold": "Ti-3d", "value": 3.0}],
        "units": [
            {"id": "u1", "calcType": "scf", "params": {
                "tstress": False, "tprnfor": True, "occupations": "fixed", "convThr": 1e-9,
                "kPoints": {"type": "automatic", "grid": [6, 6, 8], "shift": [0, 0, 0]}}},
            {"id": "u2", "calcType": "bands", "params": {
                "occupations": "fixed", "convThr": 1e-10,
                "kPoints": {"type": "crystal_b", "path": path}}},
            {"id": "u3", "calcType": "dos", "params": {}},
            {"id": "u4", "calcType": "charge", "name": "rho", "params": {"plotNum": 0}},
        ],
    })
    assert r.status_code == 200
    units = {u["id"]: u for u in r.json()["units"]}
    assert units["u1"]["executable"] == "pw.x"
    assert "ibrav       = 6" in units["u1"]["input"] and "HUBBARD" in units["u1"]["input"]
    assert units["u2"]["executable"] == "pw.x" and "K_POINTS {crystal_b}" in units["u2"]["input"]
    assert units["u3"]["executable"] == "dos.x" and "&DOS" in units["u3"]["input"]
    assert units["u4"]["executable"] == "pp.x" and "&INPUTPP" in units["u4"]["input"]


def test_generate_pw_without_structure_400():
    r = client.post("/dft/generate", json={
        "prefix": "x", "ecutwfc": 1, "ecutrho": 1,
        "units": [{"id": "z", "calcType": "scf", "params": {}}],
    })
    assert r.status_code == 400


# ── /dft/submit + /dft/advance (mock io/advance/get_job_labels) ──────────────
import json as _json  # noqa: E402
import os as _os  # noqa: E402

import src.dft.routes as routes  # noqa: E402

_SI_WF = _json.load(open(_os.path.join(_os.path.dirname(__file__), "fixtures", "si_workflow.json")))


class _FakeIO:
    def __init__(self):
        self.created = []

    def create_workflow(self, t, w, wf):
        self.created.append((t, w, wf))


def test_submit_persists_and_launches(monkeypatch):
    fake = _FakeIO()
    calls = []
    monkeypatch.setattr(routes, "_dft_io", lambda: fake)
    monkeypatch.setattr(routes, "advance", lambda io, t, w, e: (calls.append((t, w, e)) or "running"))
    r = client.post("/dft/submit", json={
        "tenantId": "t1", "workflowId": "w1",
        "workflow": {"structure": _SI_WF["structure"], "global": _SI_WF["global"], "units": _SI_WF["units"]},
    })
    assert r.status_code == 200
    assert r.json() == {"workflowId": "w1", "overallStatus": "running"}
    assert fake.created and fake.created[0][:2] == ("t1", "w1")
    assert calls == [("t1", "w1", None)]  # advance(start)


def test_submit_rejects_bad_workflow():
    r = client.post("/dft/submit", json={"tenantId": "t1", "workflowId": "w1", "workflow": {"units": []}})
    assert r.status_code == 400  # missing 'structure'


def _pubsub_envelope(job_state, job_name="projects/p/locations/r/jobs/dft-w1-relax", mtype="JOB_STATE_CHANGED"):
    return {"message": {"attributes": {"Type": mtype, "NewJobState": job_state, "JobName": job_name}}}


def test_advance_succeeded_maps_labels_and_advances(monkeypatch):
    calls = []
    monkeypatch.setattr(routes, "get_job_labels",
                        lambda name: {"dft_tenant": "t1", "dft_workflow": "w1", "dft_unit": "relax"})
    monkeypatch.setattr(routes, "_dft_io", lambda: object())
    monkeypatch.setattr(routes, "advance", lambda io, t, w, e: calls.append((t, w, e)) or "running")
    r = client.post("/dft/advance", json=_pubsub_envelope("SUCCEEDED"))
    assert r.status_code == 204
    assert calls == [("t1", "w1", {"unitId": "relax", "state": "SUCCEEDED"})]


def test_advance_ignores_intermediate_state(monkeypatch):
    calls = []
    monkeypatch.setattr(routes, "advance", lambda *a: calls.append(a))
    r = client.post("/dft/advance", json=_pubsub_envelope("RUNNING"))
    assert r.status_code == 204
    assert calls == []  # no-op tick


def test_advance_missing_labels_acks(monkeypatch):
    calls = []
    monkeypatch.setattr(routes, "get_job_labels", lambda name: {})  # not ours
    monkeypatch.setattr(routes, "_dft_io", lambda: object())
    monkeypatch.setattr(routes, "advance", lambda *a: calls.append(a))
    r = client.post("/dft/advance", json=_pubsub_envelope("FAILED"))
    assert r.status_code == 204
    assert calls == []


def test_advance_bad_envelope():
    r = client.post("/dft/advance", json={"no_message": True})
    assert r.status_code == 400
