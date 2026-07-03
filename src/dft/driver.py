"""
driver.py — event-driven DFT workflow driver.

Unlike the papers pipeline (synchronous within one Pub/Sub push), each DFT unit is
a Cloud Batch job running minutes-to-hours — far beyond a Cloud Run request. So the
DAG advances by *ticks*: each advance() call applies one state change (a unit's
Batch job finished) and launches whatever became runnable, then returns. Batch
job-state-change notifications (→ Pub/Sub → /dft/advance) drive subsequent ticks
until the workflow is terminal.

advance() is pure orchestration over an injected DftIO (Firestore + GCS + Batch +
generator + parser all live behind it), so the driver is unit-testable with a fake
IO and no GCP.

Restart wiring:
  - relax/vc-relax → downstream is a STRUCTURE handoff (parse final coords →
    relaxed_structure_from_out → used for descendants' .in generation).
  - scf/nscf → downstream restarts from the OUTDIR (.save); those become gcs_deps so
    the container stages the charge density/wavefunctions before running.

@phase R272w-h (DFT P1-3 — driver core)
"""
from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from src.dft.orchestrator import WorkflowState, is_relax, relaxed_structure_from_out
from src.dft.qe_parser import summarize_results

logger = logging.getLogger(__name__)


class DftIO(Protocol):
    """I/O surface the driver delegates to (real impl wires Firestore + GCS + Batch)."""

    def load(self, tenant_id: str, workflow_id: str) -> dict[str, Any]:
        """Return {units, structure, global, snapshot, relaxedStructures}."""
        ...

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
        """Generate .in + upload to GCS + submit a Batch job; return the job name."""
        ...

    def fetch_output(self, tenant_id: str, workflow_id: str, unit_id: str) -> str:
        """Download the unit's QE .out text from GCS."""
        ...

    def batch_job_name(self, workflow_id: str, unit_id: str) -> str:
        """Fully-qualified Batch job resource name for a unit (for state queries)."""
        ...

    def job_state(self, job_name: str) -> str:
        """Batch job state name, or 'NOT_FOUND' if the job no longer exists."""
        ...

    def save(
        self,
        tenant_id: str,
        workflow_id: str,
        snapshot: dict[str, Any],
        overall: str,
        relaxed_structures: dict[str, Any],
    ) -> None:
        """Persist the unit-status snapshot + overall status + relaxed structures."""
        ...


def advance(
    io: DftIO,
    tenant_id: str,
    workflow_id: str,
    event: dict[str, Any] | None = None,
) -> str:
    """Advance a workflow by one tick.

    ``event`` is None for the initial start, or {"unitId": str,
    "state": "SUCCEEDED"|"FAILED"|...} from a Batch notification. Returns the
    workflow overall status after the tick.
    """
    doc = io.load(tenant_id, workflow_id)
    ws = WorkflowState(doc["units"])
    if doc.get("snapshot"):
        ws.load_snapshot(doc["snapshot"])
    relaxed: dict[str, Any] = dict(doc.get("relaxedStructures") or {})
    base_structure: dict[str, Any] = doc["structure"]
    global_params: dict[str, Any] = doc.get("global") or {}

    if event:
        _apply_event(io, ws, relaxed, base_structure, tenant_id, workflow_id, event)

    for uid in ws.next_runnable():
        calc = ws.calc_type(uid) or ""
        structure = _structure_for(ws, relaxed, base_structure, uid)
        gcs_deps = [d for d in ws.depends_on(uid) if not is_relax(ws.calc_type(d))]
        job = io.launch(tenant_id, workflow_id, uid, calc, structure, global_params, gcs_deps)
        ws.mark_queued(uid)
        logger.info("dft.advance launched unit=%s calc=%s job=%s deps=%s", uid, calc, job, gcs_deps)

    overall = ws.overall_status()
    results = None
    if overall == "completed":
        try:
            results = _summarize_completed(io, ws, tenant_id, workflow_id)
            logger.info(
                "dft.advance results workflow=%s gap=%s",
                workflow_id, (results or {}).get("bandGap"),
            )
        except Exception as exc:  # noqa: BLE001 — results extraction must not break the DAG
            logger.warning("dft.advance summarize failed workflow=%s: %s", workflow_id, exc)
    io.save(tenant_id, workflow_id, ws.snapshot(), overall, relaxed, results)
    logger.info("dft.advance workflow=%s overall=%s", workflow_id, overall)
    return overall


# QUEUED longer than this (no VM ever provisioned) is treated as stuck — almost
# always an unmet regional quota (e.g. C2_CPUS) or a capacity shortfall.
_STUCK_QUEUED_SECONDS = 25 * 60


def reconcile(
    io: DftIO,
    tenant_id: str,
    workflow_id: str,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    """Actively poll Batch for units the DAG believes are QUEUED/RUNNING but that
    have gone silent (event-driven /dft/advance never fires for a job that stays
    QUEUED forever, or that Batch garbage-collects). Fails units whose job has
    vanished (NOT_FOUND) or that have been QUEUED past the stuck threshold, so the
    workflow surfaces 'failed: …' instead of spinning on 'running' indefinitely.

    Idempotent + read-mostly: SUCCEEDED/FAILED discovered here are applied via the
    normal event path. Returns {overall, changed:[{unitId,state,reason}]}.
    """
    now = now if now is not None else time.time()
    doc = io.load(tenant_id, workflow_id)
    ws = WorkflowState(doc["units"])
    if doc.get("snapshot"):
        ws.load_snapshot(doc["snapshot"])
    snap = doc.get("snapshot") or {}

    changed: list[dict[str, Any]] = []
    for uid, st in ws.snapshot().items():
        status = st.get("status")
        if status not in ("queued", "running"):
            continue
        try:
            state = io.job_state(io.batch_job_name(workflow_id, uid))
        except Exception as exc:  # noqa: BLE001 — never let one bad unit stall the sweep
            logger.warning("dft.reconcile state query failed unit=%s: %s", uid, exc)
            continue

        if state in ("SUCCEEDED", "FAILED"):
            _apply_event(io, ws, dict(doc.get("relaxedStructures") or {}),
                         doc["structure"], tenant_id, workflow_id,
                         {"unitId": uid, "state": state})
            changed.append({"unitId": uid, "state": state, "reason": f"batch state {state}"})
        elif state == "NOT_FOUND":
            ws.mark_failed(uid, f"Batch job for unit {uid!r} no longer exists (deleted or expired).")
            changed.append({"unitId": uid, "state": "FAILED", "reason": "job vanished"})
        elif state in ("QUEUED", "SCHEDULED"):
            started = snap.get(uid, {}).get("startedAt")
            if started and (now - float(started)) > _STUCK_QUEUED_SECONDS:
                mins = int((now - float(started)) / 60)
                ws.mark_failed(
                    uid,
                    f"Stuck in QUEUED for ~{mins} min — the machine could not be provisioned "
                    f"(regional quota such as C2_CPUS too low, or no capacity). "
                    f"Try a smaller preset or a family with available quota.",
                )
                changed.append({"unitId": uid, "state": "FAILED", "reason": "stuck: quota/capacity"})
        # RUNNING within threshold, or freshly QUEUED → leave alone

    overall = ws.overall_status()
    if changed:
        io.save(tenant_id, workflow_id, ws.snapshot(), overall,
                dict(doc.get("relaxedStructures") or {}), None)
        logger.info("dft.reconcile workflow=%s overall=%s changed=%d",
                    workflow_id, overall, len(changed))
    return {"overall": overall, "changed": changed}


def _summarize_completed(
    io: DftIO,
    ws: WorkflowState,
    tenant_id: str,
    workflow_id: str,
) -> dict[str, Any]:
    """On completion: fetch key unit .out files by calc type → structured scientific results."""
    by_calc: dict[str, str] = {}
    for uid in ws.snapshot():
        ct = (ws.calc_type(uid) or "").lower()
        if ct in ("vc-relax", "relax", "scf", "nscf", "bands"):
            try:
                by_calc[ct] = io.fetch_output(tenant_id, workflow_id, uid)
            except Exception as exc:  # noqa: BLE001
                logger.warning("dft.summarize fetch failed unit=%s: %s", uid, exc)
    return summarize_results(by_calc)


def _apply_event(
    io: DftIO,
    ws: WorkflowState,
    relaxed: dict[str, Any],
    base_structure: dict[str, Any],
    tenant_id: str,
    workflow_id: str,
    event: dict[str, Any],
) -> None:
    uid = event.get("unitId")
    state = event.get("state")
    if uid is None or uid not in ws.snapshot():
        return
    if state == "SUCCEEDED":
        ws.mark_completed(uid)
        if is_relax(ws.calc_type(uid)):
            out = io.fetch_output(tenant_id, workflow_id, uid)
            rs = relaxed_structure_from_out(out, base_structure)
            if rs is not None:
                relaxed[uid] = rs
                logger.info("dft.advance handoff: relaxed structure from unit=%s", uid)
    elif state == "FAILED":
        ws.mark_failed(uid, f"batch job for unit {uid!r} failed")
    # QUEUED/RUNNING/other → no-op tick


def _structure_for(
    ws: WorkflowState,
    relaxed: dict[str, Any],
    base_structure: dict[str, Any],
    uid: str,
) -> dict[str, Any]:
    """Use the relaxed structure of a transitive relax ancestor if one is available,
    else the workflow's input structure."""
    for aid in ws.ancestors(uid):
        if aid in relaxed:
            return relaxed[aid]
    return base_structure
