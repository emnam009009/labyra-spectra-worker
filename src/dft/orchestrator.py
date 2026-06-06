"""
orchestrator.py — pure DAG state machine for DFT workflow execution.

A DftWorkflow is a DAG of units (DftUnit.dependsOn edges). This module is the
PLANNER: given the units and their current statuses it decides what may run next,
applies state transitions, propagates failure down a branch (§9.6 "fail → mark
failed, stop branch"), derives the overall status, and computes the relax→scf
structure handoff. It does NO I/O — the driver (Pub/Sub trigger + GCS +
batch_client, P1-3) owns submission/parsing and calls back into this state
machine. Keeping the state machine pure makes the DAG logic unit-testable
without GCP.

Status vocabulary matches the app's DftUnitStatus:
  pending → queued → running → completed | failed

@phase R272w-f (DFT P1 — orchestrator state machine)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from src.dft.qe_parser import parse_final_structure

# DftUnitStatus (matches src/types/dft.ts)
PENDING = "pending"
QUEUED = "queued"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
_TERMINAL = (COMPLETED, FAILED)

_RELAX_CALCS = {"relax", "vc-relax"}


@dataclass
class UnitState:
    status: str = PENDING
    started_at: float | None = None
    finished_at: float | None = None
    error_message: str | None = None


@dataclass
class _Unit:
    id: str
    calc_type: str | None
    depends_on: tuple[str, ...]


class WorkflowState:
    """Holds the DAG + per-unit status; exposes the scheduling state machine."""

    def __init__(self, units: list[dict[str, Any]], *, now: Callable[[], float] = time.time):
        self._now = now
        self._units: dict[str, _Unit] = {}
        self._order: list[str] = []
        for u in units:
            uid = u["id"]
            if uid in self._units:
                raise ValueError(f"duplicate unit id {uid!r}")
            self._units[uid] = _Unit(
                id=uid,
                calc_type=u.get("calcType") or u.get("calc_type"),
                depends_on=tuple(u.get("dependsOn") or u.get("depends_on") or []),
            )
            self._order.append(uid)
        self._validate()
        self.states: dict[str, UnitState] = {uid: UnitState() for uid in self._order}

    # ── validation: missing deps + cycles ──
    def _validate(self) -> None:
        for u in self._units.values():
            for dep in u.depends_on:
                if dep not in self._units:
                    raise ValueError(f"unit {u.id!r} depends on unknown unit {dep!r}")
        indeg = {uid: len(self._units[uid].depends_on) for uid in self._units}
        queue = [uid for uid, d in indeg.items() if d == 0]
        seen = 0
        while queue:
            cur = queue.pop()
            seen += 1
            for u in self._units.values():
                if cur in u.depends_on:
                    indeg[u.id] -= 1
                    if indeg[u.id] == 0:
                        queue.append(u.id)
        if seen != len(self._units):
            raise ValueError("workflow DAG has a cycle")

    def calc_type(self, uid: str) -> str | None:
        return self._units[uid].calc_type

    def depends_on(self, uid: str) -> tuple[str, ...]:
        return self._units[uid].depends_on

    def ancestors(self, uid: str) -> set[str]:
        """All transitive dependencies of ``uid`` (walks dependsOn upward)."""
        seen: set[str] = set()
        stack = list(self._units[uid].depends_on)
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(self._units[cur].depends_on)
        return seen

    # ── scheduling ──
    def next_runnable(self) -> list[str]:
        """Unit ids that may start now: pending with all deps completed."""
        out = []
        for uid in self._order:
            if self.states[uid].status != PENDING:
                continue
            if all(self.states[d].status == COMPLETED for d in self._units[uid].depends_on):
                out.append(uid)
        return out

    def mark_queued(self, uid: str) -> None:
        self.states[uid].status = QUEUED

    def mark_running(self, uid: str) -> None:
        st = self.states[uid]
        st.status = RUNNING
        st.started_at = self._now()

    def mark_completed(self, uid: str) -> None:
        st = self.states[uid]
        st.status = COMPLETED
        st.finished_at = self._now()

    def mark_failed(self, uid: str, error: str) -> list[str]:
        """Mark a unit failed and propagate to all transitive dependents (stop the
        branch, §9.6). Returns the unit ids newly marked failed (incl. uid)."""
        st = self.states[uid]
        st.status = FAILED
        st.error_message = error
        st.finished_at = self._now()
        newly = [uid]
        changed = True
        while changed:
            changed = False
            for u in self._units.values():
                cur = self.states[u.id]
                if cur.status in _TERMINAL:
                    continue
                failed_dep = next(
                    (d for d in u.depends_on if self.states[d].status == FAILED), None
                )
                if failed_dep is not None:
                    cur.status = FAILED
                    cur.error_message = f"blocked: upstream unit {failed_dep!r} failed"
                    cur.finished_at = self._now()
                    newly.append(u.id)
                    changed = True
        return newly

    def overall_status(self) -> str:
        sts = [s.status for s in self.states.values()]
        if any(s == FAILED for s in sts):
            return FAILED
        if sts and all(s == COMPLETED for s in sts):
            return COMPLETED
        if any(s in (RUNNING, QUEUED) for s in sts):
            return RUNNING
        return PENDING

    def is_done(self) -> bool:
        return all(s.status in _TERMINAL for s in self.states.values())

    # ── persistence (resume across Pub/Sub deliveries via Firestore) ──
    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            uid: {
                "status": s.status,
                "startedAt": s.started_at,
                "finishedAt": s.finished_at,
                "errorMessage": s.error_message,
            }
            for uid, s in self.states.items()
        }

    def load_snapshot(self, snap: dict[str, dict[str, Any]]) -> None:
        for uid, s in snap.items():
            if uid in self.states:
                self.states[uid] = UnitState(
                    status=s.get("status", PENDING),
                    started_at=s.get("startedAt"),
                    finished_at=s.get("finishedAt"),
                    error_message=s.get("errorMessage"),
                )


def is_relax(calc_type: str | None) -> bool:
    return calc_type in _RELAX_CALCS


def relaxed_structure_from_out(
    out_text: str, base_structure: dict[str, Any]
) -> dict[str, Any] | None:
    """After a relax/vc-relax completes, build the downstream DftStructure from its
    optimized geometry: ibrav=0 + cellParameters (relaxed cell, Å) + relaxed crystal
    positions, carrying species/pseudos from the input structure. Returns None if the
    output has no 'final coordinates' block (not a relax run)."""
    fs = parse_final_structure(out_text)
    if fs is None or not fs["cell_ang"] or not fs["species"]:
        return None
    return {
        "ibrav": 0,
        "cellParameters": fs["cell_ang"],
        "nat": fs["n_atoms"],
        "ntyp": len(set(fs["species"])),
        "atomicSpecies": base_structure.get("atomicSpecies", []),
        "atomicPositions": [
            {"element": sp, "x": p[0], "y": p[1], "z": p[2]}
            for sp, p in zip(fs["species"], fs["frac_positions"])
        ],
        "positionsType": "crystal",
    }
