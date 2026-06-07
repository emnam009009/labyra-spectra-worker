"""
batch_client.py — Google Cloud Batch backend for DFT execution.

Two compute backends exist (§9.6): **generate-only** (default — render .in → GCS →
user runs QE on their own machine → uploads output) and **cloud-batch** (this file —
1-click: provision a Spot VM, run the QE Docker container, output to GCS, auto-delete
the VM). Batch is opt-in per job, never the default.

The manifest builder (`build_batch_job`, `qe_command`, `MACHINE_PRESETS`) is PURE —
no SDK import — and is unit-tested + schema-validated against `batch_v1.Job`. The
submit/poll/cancel wrappers lazy-import the SDK (mocked in tests, real GCP at runtime).

Cost guardrails are MANDATORY (DFT jobs are heavy): `maxRunDuration` (kill runaway
SCF), `lifecyclePolicies` FAIL_TASK on exit codes [1,137,139] (1=error, 137=OOM,
139=SIGSEGV), `maxRetryCount=0` (no auto-retry on scientific failure — burns money),
and Spot VMs. Budget cap lives at the tenant layer.

⚠ Verify region GPU quota + Spot availability + Batch API version at deploy
  (the LifecyclePolicy.Action enum is FAIL_TASK, NOT the FAIL_JOB shown in some docs).

IAM: service account with roles/batch.jobsEditor (least privilege).

@phase R272w-e (DFT P1 — Cloud Batch execution)
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Machine presets — combo, not raw cpu/mem (§9.6). L4 GPU attaches to the g2 family.
MACHINE_PRESETS: dict[str, dict[str, Any]] = {
    # `vcpu` drives NPROC (MPI ranks). machineType-based presets pin a compute-optimized
    # family (better QE perf than the e2 Batch auto-picks from cpuMilli) and omit
    # computeResource (the machine fully defines CPU/RAM).
    "low": {"vcpu": 4, "cpuMilli": 4000, "memoryMib": 16384, "gpu": None, "machineType": None},
    "standard": {"vcpu": 8, "cpuMilli": 8000, "memoryMib": 32768, "gpu": None, "machineType": None},
    # bulk = default for real workloads: c2-standard-60 (Intel Cascade Lake, AVX-512, 30
    # physical cores → np=30, 2×3×5 factors cleanly for npool). 120GB... actually 240GB RAM.
    "bulk": {"vcpu": 60, "cpuMilli": None, "memoryMib": None, "gpu": None, "machineType": "c2-standard-60"},
    # fallback if c2 capacity-constrained in the zone (c2 is older-gen; on-demand may stall).
    "bulk-amd": {"vcpu": 16, "cpuMilli": None, "memoryMib": None, "gpu": None, "machineType": "c2d-standard-16"},
    "bulk-large": {"vcpu": 32, "cpuMilli": None, "memoryMib": None, "gpu": None, "machineType": "c2d-standard-32"},
    # ⚠ GPU is a knob only: the QE image is CPU-only → GPU sits idle until a CUDA QE build exists.
    "high-gpu": {"vcpu": 8, "cpuMilli": None, "memoryMib": None, "gpu": "nvidia-l4", "machineType": "g2-standard-8"},
}


def preset_vcpu(machine_preset: str) -> int:
    """vCPU count of a preset → default NPROC (MPI ranks)."""
    if machine_preset not in MACHINE_PRESETS:
        raise ValueError(f"unknown machine preset {machine_preset!r}")
    return int(MACHINE_PRESETS[machine_preset]["vcpu"])

# Fail (do NOT retry) on hard errors: 1=generic, 137=OOM (SIGKILL), 139=SIGSEGV.
_FAIL_EXIT_CODES = [1, 137, 139]

# QE binary per calc type.
_QE_BINARY: dict[str, str] = {
    "vc-relax": "pw.x", "relax": "pw.x", "scf": "pw.x", "nscf": "pw.x", "bands": "pw.x",
    "ppbands": "bands.x", "dos": "dos.x", "pdos": "projwfc.x", "charge": "pp.x",
}


def qe_command(calc_type: str, in_file: str, out_file: str | None = None) -> list[str]:
    """Container command for a QE calc. QE writes to stdout; when ``out_file`` is given
    we wrap in a shell so the .out lands in the (GCS-mounted) work dir for parsing."""
    binary = _QE_BINARY.get(calc_type)
    if binary is None:
        raise ValueError(f"no QE binary for calc type {calc_type!r}")
    if out_file:
        return ["bash", "-c", f"{binary} -in {in_file} > {out_file} 2>&1"]
    return [binary, "-in", in_file]


def build_batch_job(
    image_uri: str,
    commands: list[str],
    *,
    machine_preset: str = "standard",
    max_run_duration_sec: int = 3600,
    use_spot: bool = True,
    env: dict[str, str] | None = None,
    gpu_count: int = 1,
    labels: dict[str, str] | None = None,
    notifications_topic: str | None = None,
    service_account: str | None = None,
) -> dict[str, Any]:
    """Build a guardrailed Cloud Batch job manifest (dict).

    Pure + schema-validated against ``batch_v1.Job`` in tests. Guardrails baked in:
    maxRunDuration, FAIL_TASK on [1,137,139], maxRetryCount=0, Spot (default).
    """
    if machine_preset not in MACHINE_PRESETS:
        raise ValueError(
            f"unknown machine preset {machine_preset!r} (want one of {sorted(MACHINE_PRESETS)})"
        )
    preset = MACHINE_PRESETS[machine_preset]

    container: dict[str, Any] = {"imageUri": image_uri}
    if commands:  # omit for ENTRYPOINT-driven images (QE image reads env, runs entrypoint)
        container["commands"] = list(commands)
    runnable: dict[str, Any] = {"container": container}
    if env:
        runnable["environment"] = {"variables": dict(env)}

    task_spec: dict[str, Any] = {
        "runnables": [runnable],
        "maxRunDuration": f"{int(max_run_duration_sec)}s",
        "maxRetryCount": 0,
        "lifecyclePolicies": [
            {"action": "FAIL_TASK", "actionCondition": {"exitCodes": list(_FAIL_EXIT_CODES)}}
        ],
    }
    if preset["machineType"] is None:  # auto-pick machine from cpu/mem (no explicit machineType)
        task_spec["computeResource"] = {"cpuMilli": preset["cpuMilli"], "memoryMib": preset["memoryMib"]}

    policy: dict[str, Any] = {"provisioningModel": "SPOT" if use_spot else "STANDARD"}
    if preset["machineType"]:  # pin compute-optimized / GPU family
        policy["machineType"] = preset["machineType"]
    instance: dict[str, Any] = {"policy": policy}
    if preset["gpu"]:
        policy["accelerators"] = [{"type": preset["gpu"], "count": gpu_count}]
        instance["installGpuDrivers"] = True

    allocation: dict[str, Any] = {"instances": [instance]}
    if service_account:  # Batch VM runs as this SA (needs storage.objectAdmin on the bucket)
        allocation["serviceAccount"] = {"email": service_account}
    job: dict[str, Any] = {
        "taskGroups": [{"taskSpec": task_spec, "taskCount": 1, "parallelism": 1}],
        "allocationPolicy": allocation,
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
    }
    if labels:
        # mapped back to (tenant, workflow, unit) via get_job on a Batch notification
        job["labels"] = dict(labels)
    if notifications_topic:
        # Batch publishes JobUID/JobName/NewJobState/Type to this topic on every job
        # state change; the /dft/advance handler acts on SUCCEEDED/FAILED, no-ops the rest.
        job["notifications"] = [
            {"pubsubTopic": notifications_topic, "message": {"type": "JOB_STATE_CHANGED"}}
        ]
    return job


# ── SDK wrappers (thin; lazy-import SDK; mocked in tests, real GCP at runtime) ─


def _client() -> Any:  # pragma: no cover - real GCP client
    from google.cloud import batch_v1

    return batch_v1.BatchServiceClient()


def submit_job(
    project: str,
    region: str,
    job_id: str,
    manifest: dict[str, Any],
    client: Any | None = None,
) -> str:
    """Create a Batch job from a manifest dict; returns the job resource name."""
    from google.cloud import batch_v1
    from google.protobuf import json_format

    client = client or _client()
    job = batch_v1.Job()
    json_format.ParseDict(manifest, job._pb)  # also validates the manifest schema
    created = client.create_job(
        parent=f"projects/{project}/locations/{region}", job=job, job_id=job_id
    )
    logger.info("submitted batch job %s", created.name)
    return created.name


def get_job_state(job_name: str, client: Any | None = None) -> str:
    """Return the Batch job state name (QUEUED/SCHEDULED/RUNNING/SUCCEEDED/FAILED/…)."""
    from google.cloud import batch_v1

    client = client or _client()
    job = client.get_job(name=job_name)
    return batch_v1.JobStatus.State(job.status.state).name


def get_job_labels(job_name: str, client: Any | None = None) -> dict[str, str]:
    """Read a Batch job's labels (the /dft/advance handler maps JobName →
    {dft_tenant, dft_workflow, dft_unit}, since the notification carries only JobName)."""
    from google.cloud import batch_v1

    client = client or _client()
    job = client.get_job(name=job_name)
    return dict(job.labels)


def cancel_job(job_name: str, client: Any | None = None) -> None:
    """Request cancellation of a running Batch job (the [Stop job] action)."""
    client = client or _client()
    client.cancel_job(name=job_name)
    logger.info("cancelled batch job %s", job_name)
