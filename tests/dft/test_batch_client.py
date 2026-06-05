"""
Tests for dft.batch_client — Cloud Batch manifest builder + SDK wrappers.

Pure tests (manifest structure, qe_command) always run. Schema-validation +
submit/poll/cancel tests importorskip the SDK (run wherever google-cloud-batch
is installed; submit/poll/cancel use a mock client — no real GCP).

@phase R272w-e (DFT P1)
"""
from unittest.mock import MagicMock

import pytest

from src.dft.batch_client import (
    MACHINE_PRESETS,
    build_batch_job,
    cancel_job,
    get_job_state,
    qe_command,
    submit_job,
)

IMAGE = "gcr.io/proj/quantum-espresso:7.4.1"


# ── pure: manifest structure ─────────────────────────────────────────────────


def test_standard_manifest_guardrails():
    job = build_batch_job(IMAGE, ["pw.x", "-in", "pw_scf.in"], machine_preset="standard")
    ts = job["taskGroups"][0]["taskSpec"]
    assert ts["computeResource"] == {"cpuMilli": 8000, "memoryMib": 32768}
    assert ts["maxRunDuration"] == "3600s"
    assert ts["maxRetryCount"] == 0  # no auto-retry on scientific failure
    lp = ts["lifecyclePolicies"][0]
    assert lp["action"] == "FAIL_TASK"  # NOT FAIL_JOB (invalid enum)
    assert lp["actionCondition"]["exitCodes"] == [1, 137, 139]
    inst = job["allocationPolicy"]["instances"][0]
    assert inst["policy"]["provisioningModel"] == "SPOT"
    assert "accelerators" not in inst["policy"]
    assert "installGpuDrivers" not in inst
    assert ts["runnables"][0]["container"] == {"imageUri": IMAGE, "commands": ["pw.x", "-in", "pw_scf.in"]}
    assert job["logsPolicy"]["destination"] == "CLOUD_LOGGING"


def test_low_preset_and_no_spot():
    job = build_batch_job(IMAGE, ["pw.x"], machine_preset="low", use_spot=False,
                          max_run_duration_sec=1800)
    ts = job["taskGroups"][0]["taskSpec"]
    assert ts["computeResource"] == {"cpuMilli": 4000, "memoryMib": 16384}
    assert ts["maxRunDuration"] == "1800s"
    assert job["allocationPolicy"]["instances"][0]["policy"]["provisioningModel"] == "STANDARD"


def test_high_gpu_attaches_l4():
    job = build_batch_job(IMAGE, ["pw.x"], machine_preset="high-gpu", gpu_count=1)
    inst = job["allocationPolicy"]["instances"][0]
    assert inst["policy"]["machineType"] == "g2-standard-8"
    assert inst["policy"]["accelerators"] == [{"type": "nvidia-l4", "count": 1}]
    assert inst["installGpuDrivers"] is True


def test_env_variables_passed():
    job = build_batch_job(IMAGE, ["pw.x"], env={"OMP_NUM_THREADS": "1"})
    runnable = job["taskGroups"][0]["taskSpec"]["runnables"][0]
    assert runnable["environment"]["variables"] == {"OMP_NUM_THREADS": "1"}


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="unknown machine preset"):
        build_batch_job(IMAGE, ["pw.x"], machine_preset="mega")


def test_presets_table():
    assert set(MACHINE_PRESETS) == {"low", "standard", "high-gpu"}
    assert MACHINE_PRESETS["high-gpu"]["gpu"] == "nvidia-l4"


# ── pure: qe_command ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "calc,binary",
    [("vc-relax", "pw.x"), ("scf", "pw.x"), ("bands", "pw.x"),
     ("ppbands", "bands.x"), ("dos", "dos.x"), ("pdos", "projwfc.x"), ("charge", "pp.x")],
)
def test_qe_command_binaries(calc, binary):
    assert qe_command(calc, "x.in")[0] == binary


def test_qe_command_redirect():
    cmd = qe_command("scf", "pw_scf.in", out_file="pw_scf.out")
    assert cmd == ["bash", "-c", "pw.x -in pw_scf.in > pw_scf.out 2>&1"]


def test_qe_command_unknown_raises():
    with pytest.raises(ValueError, match="no QE binary"):
        qe_command("md", "x.in")


# ── SDK: schema validation + wrappers (skip if SDK absent) ───────────────────


def test_manifest_validates_against_batch_proto():
    batch_v1 = pytest.importorskip("google.cloud.batch_v1")
    from google.protobuf import json_format

    for preset in MACHINE_PRESETS:
        manifest = build_batch_job(IMAGE, ["pw.x", "-in", "a.in"], machine_preset=preset)
        job = batch_v1.Job()
        json_format.ParseDict(manifest, job._pb)  # raises if any field/enum is wrong
        assert job.task_groups[0].task_spec.compute_resource.cpu_milli == \
            MACHINE_PRESETS[preset]["cpuMilli"]


def test_submit_job_calls_create_with_mock():
    pytest.importorskip("google.cloud.batch_v1")
    client = MagicMock()
    created = MagicMock()
    created.name = "projects/p/locations/r/jobs/j1"  # `.name` set explicitly (MagicMock name= kwarg is reserved)
    client.create_job.return_value = created
    manifest = build_batch_job(IMAGE, ["pw.x", "-in", "a.in"])
    name = submit_job("p", "asia-southeast1", "j1", manifest, client=client)
    assert name == "projects/p/locations/r/jobs/j1"
    _, kwargs = client.create_job.call_args
    assert kwargs["parent"] == "projects/p/locations/asia-southeast1"
    assert kwargs["job_id"] == "j1"


def test_get_job_state_maps_enum_with_mock():
    batch_v1 = pytest.importorskip("google.cloud.batch_v1")
    client = MagicMock()
    job = MagicMock()
    job.status.state = batch_v1.JobStatus.State.RUNNING
    client.get_job.return_value = job
    assert get_job_state("projects/p/locations/r/jobs/j1", client=client) == "RUNNING"


def test_cancel_job_calls_cancel_with_mock():
    pytest.importorskip("google.cloud.batch_v1")
    client = MagicMock()
    cancel_job("projects/p/locations/r/jobs/j1", client=client)
    client.cancel_job.assert_called_once_with(name="projects/p/locations/r/jobs/j1")
