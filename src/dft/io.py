"""
io.py — real DftIO: Firestore (workflow doc) + GCS (.in/.out) + generator + Batch.

Implements the dft.driver.DftIO protocol the driver delegates to. The driver stays
pure; all GCP plumbing lives here. Clients are lazy + injectable (tests pass fakes,
no GCP). The loaded workflow doc is cached so launch() can read each unit's per-calc
`params` (kPoints/occupations/…) without changing the driver's launch signature.

GCS layout (bucket = settings):
  workflows/{wid}/units/{uid}/{calc}.in   ← uploaded here; entrypoint reads it
  workflows/{wid}/units/{uid}/{calc}.out  ← QE writes here (staged back by entrypoint)
  workflows/{wid}/units/{uid}/out/        ← QE outdir (.save) for downstream restart
  pseudo/                                 ← pseudopotential UPFs (shared)

Firestore: tenants/{tid}/dftWorkflows/{wid} = {units, structure, global, snapshot,
relaxedStructures, overallStatus}.

@phase R272w-j (DFT P1-3b — real IO)
"""
from __future__ import annotations

import logging
import re
from typing import Any

from src.dft.batch_client import _QE_BINARY, build_batch_job
from src.dft.errors import FatalError
from src.dft.generator import generate_postproc_input, generate_pw_input

logger = logging.getLogger(__name__)

_POSTPROC = {"ppbands", "dos", "pdos", "charge"}
_STAGED_OUTDIR = "./out"  # MUST match entrypoint.sh staging dir


def _sanitize_job_id(raw: str) -> str:
    """Batch job id: ^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$ — lowercase, ≤63, starts alpha."""
    s = re.sub(r"[^a-z0-9-]+", "-", raw.lower()).strip("-")
    if not s or not s[0].isalpha():
        s = "j-" + s
    return s[:63].rstrip("-")


class FirestoreGcsBatchIO:
    """DftIO backed by Firestore + GCS + Cloud Batch (compute backend cloud-batch)."""

    def __init__(
        self,
        *,
        project: str,
        region: str,
        bucket: str,
        image_uri: str,
        pseudo_prefix: str = "pseudo",
        topic: str | None = None,
        service_account: str | None = None,
        machine_preset: str = "low",
        nproc: int = 1,
        prefix: str = "pwscf",
        firestore_client: Any | None = None,
        storage_client: Any | None = None,
        submit_fn: Any | None = None,
    ) -> None:
        self.project = project
        self.region = region
        self.bucket = bucket
        self.image_uri = image_uri
        self.pseudo_prefix = pseudo_prefix
        self.topic = topic
        self.service_account = service_account
        self.machine_preset = machine_preset
        self.nproc = nproc
        self.prefix = prefix
        self._fs = firestore_client
        self._gcs = storage_client
        self._submit = submit_fn
        self._cache: dict[tuple[str, str], dict[str, Any]] = {}

    # ── lazy clients ──
    def _firestore(self) -> Any:
        if self._fs is None:
            from google.cloud import firestore

            self._fs = firestore.Client(project=self.project)
        return self._fs

    def _storage(self) -> Any:
        if self._gcs is None:
            from google.cloud import storage

            self._gcs = storage.Client(project=self.project)
        return self._gcs

    def _submit_job(self, job_id: str, manifest: dict[str, Any]) -> str:
        if self._submit is not None:
            return self._submit(self.project, self.region, job_id, manifest)
        from src.dft.batch_client import submit_job

        return submit_job(self.project, self.region, job_id, manifest)

    def _doc(self, tenant: str, wf: str) -> Any:
        return (
            self._firestore()
            .collection("tenants").document(tenant)
            .collection("dftWorkflows").document(wf)
        )

    def _unit_calc(self, doc: dict[str, Any], uid: str) -> str:
        for u in doc.get("units", []):
            if u["id"] == uid:
                return u.get("calcType") or u.get("calc_type") or ""
        return ""

    # ── DftIO protocol ──
    def load(self, tenant_id: str, workflow_id: str) -> dict[str, Any]:
        snap = self._doc(tenant_id, workflow_id).get()
        if not getattr(snap, "exists", False):
            raise FatalError(f"workflow not found: {tenant_id}/{workflow_id}")
        d = snap.to_dict() or {}
        doc = {
            "units": d.get("units", []),
            "structure": d.get("structure", {}),
            "global": d.get("global", {}),
            "snapshot": d.get("snapshot", {}),
            "relaxedStructures": d.get("relaxedStructures", {}),
        }
        self._cache[(tenant_id, workflow_id)] = doc
        return doc

    def launch(
        self,
        tenant_id: str,
        workflow_id: str,
        unit_id: str,
        calc_type: str,
        structure: dict[str, Any],
        global_params: dict[str, Any],
        gcs_deps: list[str],
    ) -> str:
        doc = self._cache.get((tenant_id, workflow_id), {})
        units_by_id = {u["id"]: u for u in doc.get("units", [])}
        params = (units_by_id.get(unit_id, {}) or {}).get("params") or {}

        in_text = self._render(calc_type, structure, global_params, params)
        in_name, out_name = f"{calc_type}.in", f"{calc_type}.out"
        unit_prefix = f"workflows/{workflow_id}/units/{unit_id}"
        self._upload(f"{unit_prefix}/{in_name}", in_text)

        gcs_work = f"gs://{self.bucket}/{unit_prefix}"
        env = {
            "QE_BINARY": _QE_BINARY.get(calc_type, "pw.x"),
            "QE_IN": in_name,
            "QE_OUT": out_name,
            "GCS_WORK": gcs_work,
            "GCS_PSEUDO": f"gs://{self.bucket}/{self.pseudo_prefix}",
            "NPROC": str(self.nproc),
            "OMP_NUM_THREADS": "1",
        }
        deps = " ".join(f"gs://{self.bucket}/workflows/{workflow_id}/units/{d}" for d in gcs_deps)
        if deps:
            env["GCS_DEPS"] = deps

        manifest = build_batch_job(
            self.image_uri, [],
            machine_preset=self.machine_preset,
            env=env,
            labels={"dft_tenant": tenant_id, "dft_workflow": workflow_id, "dft_unit": unit_id},
            notifications_topic=self.topic,
            service_account=self.service_account,
        )
        job_id = _sanitize_job_id(f"dft-{workflow_id}-{unit_id}")
        name = self._submit_job(job_id, manifest)
        logger.info("dft.io launched unit=%s job_id=%s name=%s", unit_id, job_id, name)
        return name

    def fetch_output(self, tenant_id: str, workflow_id: str, unit_id: str) -> str:
        doc = self._cache.get((tenant_id, workflow_id), {})
        calc = self._unit_calc(doc, unit_id)
        path = f"workflows/{workflow_id}/units/{unit_id}/{calc}.out"
        blob = self._storage().bucket(self.bucket).blob(path)
        return blob.download_as_text()

    def save(
        self,
        tenant_id: str,
        workflow_id: str,
        snapshot: dict[str, Any],
        overall: str,
        relaxed_structures: dict[str, Any],
    ) -> None:
        payload: dict[str, Any] = {
            "snapshot": snapshot,
            "overallStatus": overall,
            "relaxedStructures": relaxed_structures,
        }
        try:
            from google.cloud.firestore_v1 import SERVER_TIMESTAMP

            payload["updatedAt"] = SERVER_TIMESTAMP
        except ImportError:  # firestore not in env (tests) — skip the server timestamp
            pass
        self._doc(tenant_id, workflow_id).set(payload, merge=True)

    # ── helpers ──
    def _render(
        self, calc: str, structure: dict[str, Any], g: dict[str, Any], params: dict[str, Any]
    ) -> str:
        functional = g.get("functional", "pbe")
        if calc in _POSTPROC:
            return generate_postproc_input(
                calc, params, prefix=self.prefix, functional=functional,
                outdir=_STAGED_OUTDIR, name=params.get("name"),
            )
        ecutwfc = float(g.get("ecutwfc", 50.0))
        ecutrho = float(g.get("ecutrho", ecutwfc * 4.0))
        return generate_pw_input(
            structure, params, prefix=self.prefix, ecutwfc=ecutwfc, ecutrho=ecutrho,
            functional=functional, hubbard=g.get("hubbard"), outdir=_STAGED_OUTDIR,
        )

    def _upload(self, path: str, text: str) -> None:
        self._storage().bucket(self.bucket).blob(path).upload_from_string(text)
