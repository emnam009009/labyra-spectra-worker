"""
schema.py — request/response models for the DFT generate endpoints (Phase 0,
generate-only). The GenerateRequest mirrors the app's DftWorkflow (§5.1): a
structure + global params + a list of units (DAG nodes). No execution fields here.

@phase R272w-c (DFT P0 — endpoints)
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class StructureRequest(BaseModel):
    """Build a DftStructure from one of three sources (one gateway, verified)."""

    source: Literal["cif", "poscar", "mp_id"]
    cif_text: str | None = None
    poscar_text: str | None = None
    mp_id: str | None = None
    mp_api_key: str | None = None  # per-user MP key (source=mp_id)
    pseudo_map: dict[str, str] = {}
    use_primitive: bool = True
    prefer_ibrav: bool = True


class HubbardItem(BaseModel):
    manifold: str  # "W-5d"
    value: float  # 6.2


class UnitRequest(BaseModel):
    """One DAG node. calcType ∈ vc-relax|relax|scf|nscf|bands|ppbands|dos|pdos|charge."""

    id: str
    calcType: str
    params: dict[str, Any] = {}  # per-calc (kPoints, occupations, convThr, …)
    name: str | None = None  # charge plot label
    outdir: str | None = None


class GenerateRequest(BaseModel):
    structure: dict[str, Any] | None = None  # DftStructure (required for pw.x units)
    prefix: str
    functional: str = "pbe"
    ecutwfc: float
    ecutrho: float
    hubbard: list[HubbardItem] = []
    units: list[UnitRequest]


class GeneratedUnit(BaseModel):
    id: str
    calcType: str
    executable: str | None
    input: str


class GenerateResponse(BaseModel):
    units: list[GeneratedUnit]
