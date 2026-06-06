"""
Tests for dft.io.FirestoreGcsBatchIO — render + GCS paths + Batch env/labels, with
MagicMock Firestore/Storage and a recording submit_fn (no GCP).

@phase R272w-j (DFT P1-3b)
"""
import json
import os
from unittest.mock import MagicMock

import pytest

from src.dft.io import FirestoreGcsBatchIO, _sanitize_job_id

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
SI_WF = json.load(open(os.path.join(FIXTURES, "si_workflow.json")))
RELAX_OUT = open(os.path.join(FIXTURES, "synthetic_relax.out")).read()


def _io(exists=True, to_dict=None, download=RELAX_OUT):
    fs = MagicMock()
    doc_ref = fs.collection.return_value.document.return_value.collection.return_value.document.return_value
    doc_ref.get.return_value.exists = exists
    doc_ref.get.return_value.to_dict.return_value = to_dict if to_dict is not None else SI_WF
    gcs = MagicMock()
    gcs.bucket.return_value.blob.return_value.download_as_text.return_value = download
    submitted = []

    def submit_fn(project, region, job_id, manifest):
        submitted.append((job_id, manifest))
        return f"projects/{project}/locations/{region}/jobs/{job_id}"

    io = FirestoreGcsBatchIO(
        project="labyra-app-dev", region="asia-southeast1", bucket="labyra-app-dev-dft",
        image_uri="img:7.4.1", topic="projects/labyra-app-dev/topics/dft-advance",
        service_account="spectra-worker@labyra-app-dev.iam.gserviceaccount.com",
        firestore_client=fs, storage_client=gcs, submit_fn=submit_fn,
    )
    return io, fs, gcs, submitted, doc_ref


def test_sanitize_job_id():
    assert _sanitize_job_id("dft-WF_abc-scf") == "dft-wf-abc-scf"
    assert _sanitize_job_id("123-x").startswith("j-")
    assert len(_sanitize_job_id("a" * 100)) <= 63


def test_load_returns_doc_and_caches():
    io, *_ = _io()
    doc = io.load("t1", "w1")
    assert [u["id"] for u in doc["units"]] == ["relax", "scf", "bands"]
    assert doc["structure"]["ibrav"] == 0


def test_load_missing_raises_fatal():
    from src.dft.errors import FatalError
    io, *_ = _io(exists=False)
    with pytest.raises(FatalError, match="workflow not found"):
        io.load("t1", "w1")


def test_launch_scf_renders_uploads_submits():
    io, _fs, gcs, submitted, _ = _io()
    io.load("t1", "w1")  # populate cache so per-unit params are available
    name = io.launch("t1", "w1", "scf", "scf", SI_WF["structure"], SI_WF["global"], [])
    # upload path + a real .in rendered through the generator
    blob_paths = [c.args[0] for c in gcs.bucket.return_value.blob.call_args_list]
    assert "workflows/w1/units/scf/scf.in" in blob_paths
    in_text = gcs.bucket.return_value.blob.return_value.upload_from_string.call_args.args[0]
    assert "calculation" in in_text and "scf" in in_text and "Si" in in_text
    # submitted manifest
    job_id, manifest = submitted[0]
    assert job_id == "dft-w1-scf"
    env = manifest["taskGroups"][0]["taskSpec"]["runnables"][0]["environment"]["variables"]
    assert env["QE_BINARY"] == "pw.x" and env["QE_IN"] == "scf.in" and env["QE_OUT"] == "scf.out"
    assert env["GCS_WORK"] == "gs://labyra-app-dev-dft/workflows/w1/units/scf"
    assert env["GCS_PSEUDO"] == "gs://labyra-app-dev-dft/pseudo"
    assert "GCS_DEPS" not in env  # scf depends on relax (structure handoff) → no outdir dep
    assert manifest["labels"] == {"dft_tenant": "t1", "dft_workflow": "w1", "dft_unit": "scf"}
    assert manifest["notifications"][0]["pubsubTopic"].endswith("/dft-advance")
    assert manifest["allocationPolicy"]["serviceAccount"]["email"].startswith("spectra-worker@")
    assert name.endswith("/jobs/dft-w1-scf")


def test_launch_bands_sets_outdir_dep():
    io, _fs, _gcs, submitted, _ = _io()
    io.load("t1", "w1")
    io.launch("t1", "w1", "bands", "bands", SI_WF["structure"], SI_WF["global"], ["scf"])
    _job_id, manifest = submitted[0]
    env = manifest["taskGroups"][0]["taskSpec"]["runnables"][0]["environment"]["variables"]
    assert env["GCS_DEPS"] == "gs://labyra-app-dev-dft/workflows/w1/units/scf"


def test_fetch_output_downloads_calc_out():
    io, _fs, gcs, _s, _d = _io(download=RELAX_OUT)
    io.load("t1", "w1")
    out = io.fetch_output("t1", "w1", "relax")
    blob_paths = [c.args[0] for c in gcs.bucket.return_value.blob.call_args_list]
    assert "workflows/w1/units/relax/vc-relax.out" in blob_paths
    assert "JOB DONE" in out


def test_save_writes_merge():
    io, _fs, _gcs, _s, doc_ref = _io()
    io.save("t1", "w1", {"relax": {"status": "completed"}}, "running", {"relax": {"ibrav": 0}})
    _args, kwargs = doc_ref.set.call_args
    assert kwargs.get("merge") is True
    payload = doc_ref.set.call_args.args[0]
    assert payload["overallStatus"] == "running"
    assert payload["snapshot"]["relax"]["status"] == "completed"


def test_create_workflow_json_strings_structure():
    io, _fs, _gcs, _s, doc_ref = _io()
    io.create_workflow("t1", "w1", SI_WF)
    payload = doc_ref.set.call_args.args[0]
    assert isinstance(payload["structure"], str)  # nested-array-safe for Firestore
    assert json.loads(payload["structure"])["ibrav"] == 0
    assert payload["relaxedStructures"] == "{}"
    assert payload["units"] == SI_WF["units"]  # units stay native (arrays of maps OK)


def test_load_decodes_json_string_structure():
    enc = {
        "units": SI_WF["units"],
        "structure": json.dumps(SI_WF["structure"]),
        "global": SI_WF["global"],
        "snapshot": {"relax": {"status": "completed"}},
        "relaxedStructures": json.dumps({"relax": {"ibrav": 0, "cellParameters": [[1, 0, 0]]}}),
    }
    io, *_ = _io(to_dict=enc)
    doc = io.load("t1", "w1")
    assert doc["structure"]["ibrav"] == 0  # decoded back to dict
    assert doc["relaxedStructures"]["relax"]["cellParameters"] == [[1, 0, 0]]


def test_save_json_strings_relaxed_structures():
    io, _fs, _gcs, _s, doc_ref = _io()
    io.save("t1", "w1", {"relax": {"status": "completed"}}, "running",
            {"relax": {"ibrav": 0, "cellParameters": [[0, 2.7, 2.7]]}})
    payload = doc_ref.set.call_args.args[0]
    assert isinstance(payload["relaxedStructures"], str)
    assert json.loads(payload["relaxedStructures"])["relax"]["ibrav"] == 0


def test_launch_uses_workflow_preset_and_nproc():
    io, _fs, _gcs, submitted, _ = _io()
    io.create_workflow("t1", "w1", SI_WF, machine_preset="bulk")
    io.launch("t1", "w1", "scf", "scf", SI_WF["structure"], SI_WF["global"], [])
    _job, manifest = submitted[0]
    env = manifest["taskGroups"][0]["taskSpec"]["runnables"][0]["environment"]["variables"]
    assert env["NPROC"] == "16"  # bulk = c2d-standard-16 → 16 MPI ranks
    pol = manifest["allocationPolicy"]["instances"][0]["policy"]
    assert pol["machineType"] == "c2d-standard-16"
    assert "computeResource" not in manifest["taskGroups"][0]["taskSpec"]


def test_per_unit_preset_overrides_workflow():
    import copy
    wf = copy.deepcopy(SI_WF)
    for u in wf["units"]:
        if u["id"] == "relax":
            u["machinePreset"] = "bulk-large"  # this unit overrides workflow default
    io, _fs, _gcs, submitted, _ = _io()
    io.create_workflow("t1", "w1", wf, machine_preset="bulk")
    io.launch("t1", "w1", "relax", "vc-relax", wf["structure"], wf["global"], [])
    _job, manifest = submitted[0]
    env = manifest["taskGroups"][0]["taskSpec"]["runnables"][0]["environment"]["variables"]
    assert env["NPROC"] == "32"  # unit override bulk-large
    assert manifest["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "c2d-standard-32"


def test_max_run_sec_workflow_override():
    io, _fs, _gcs, submitted, _ = _io()
    io.create_workflow("t1", "w1", SI_WF, max_run_sec=7200)
    io.launch("t1", "w1", "scf", "scf", SI_WF["structure"], SI_WF["global"], [])
    _job, manifest = submitted[0]
    assert manifest["taskGroups"][0]["taskSpec"]["maxRunDuration"] == "7200s"
