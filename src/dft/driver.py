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
