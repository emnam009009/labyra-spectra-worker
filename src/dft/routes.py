"""
routes.py — DFT input-generator HTTP endpoints (Phase 0, generate-only).

An APIRouter (not the main app) so it imports only FastAPI + the dft module — no
heavy worker deps — which keeps it unit-testable in isolation. main.py wires it
with `app.include_router(router)`.

  POST /dft/structure  — CIF/POSCAR/MP-id → DftStructure (ibrav-verified)
  POST /dft/kpath      — structure source → seekpath BZ path (bands crystal_b)
  POST /dft/generate   — structure + units → rendered ordered QE .in files
  POST /dft/submit     — persist a DftWorkflow + launch root units (cloud-batch)
  POST /dft/advance    — Pub/Sub push (Batch JobStateChanged) → advance the DAG

@phase R272w-c (DFT P0 — endpoints)
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from src.dft import structure as dft_structure
from src.dft.generator import generate_postproc_input, generate_pw_input
from src.dft.batch_client import get_job_labels
from src.dft.driver import advance
from src.dft.errors import FatalError
from src.dft.schema import (
    GenerateRequest,
    GenerateResponse,
    StructureRequest,
    SubmitRequest,
)

router = APIRouter(prefix="/dft", tags=["dft"])

_PW_CALCS = {"vc-relax", "relax", "scf", "nscf", "bands"}
_EXECUTABLE = {
    "vc-relax": "pw.x", "relax": "pw.x", "scf": "pw.x", "nscf": "pw.x", "bands": "pw.x",
    "ppbands": "bands.x", "dos": "dos.x", "pdos": "projwfc.x", "charge": "pp.x",
}


@router.post("/structure")
def build_structure(req: StructureRequest) -> dict[str, Any]:
    """Build a DftStructure (ibrav-verified) from CIF / POSCAR / Materials Project id."""
    try:
        if req.source == "cif":
            if not req.cif_text:
                raise HTTPException(status_code=400, detail="cif_text required for source=cif")
            return dft_structure.from_cif(
                req.cif_text, req.pseudo_map,
                use_primitive=req.use_primitive, prefer_ibrav=req.prefer_ibrav,
            )
        if req.source == "poscar":
            if not req.poscar_text:
                raise HTTPException(status_code=400, detail="poscar_text required for source=poscar")
            return dft_structure.from_poscar(
                req.poscar_text, req.pseudo_map,
                use_primitive=req.use_primitive, prefer_ibrav=req.prefer_ibrav,
            )
        if not (req.mp_id and req.mp_api_key):
            raise HTTPException(status_code=400, detail="mp_id and mp_api_key required for source=mp_id")
        return dft_structure.from_mp_id(
            req.mp_id, req.mp_api_key, req.pseudo_map,
            use_primitive=req.use_primitive, prefer_ibrav=req.prefer_ibrav,
        )
    except HTTPException:
        raise
    except ValueError as exc:  # sanity failure / bad structure
        raise HTTPException(status_code=400, detail=str(exc)[:300]) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)[:300]) from exc


@router.post("/kpath")
def build_kpath(req: StructureRequest) -> dict[str, Any]:
    """High-symmetry BZ path (seekpath) for a bands K_POINTS {crystal_b} input.

    seekpath standardizes the cell — pair this path with the standardized primitive
    structure from /dft/structure (use_primitive=True), or supply a manual path.
    """
    from pymatgen.core import Structure

    from src.dft.kpath import get_kpath

    try:
        if req.source == "cif" and req.cif_text:
            st = Structure.from_str(req.cif_text, fmt="cif")
        elif req.source == "poscar" and req.poscar_text:
            st = Structure.from_str(req.poscar_text, fmt="poscar")
        elif req.source == "mp_id" and req.mp_id and req.mp_api_key:
            from mp_api.client import MPRester

            with MPRester(req.mp_api_key) as mpr:
                st = mpr.get_structure_by_material_id(req.mp_id)
        else:
            raise HTTPException(status_code=400, detail="invalid/missing structure source for kpath")
        return get_kpath(st)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)[:300]) from exc


@router.post("/generate")
def generate(req: GenerateRequest) -> GenerateResponse:
    """Render ordered QE inputs (.in) for each unit in the workflow DAG (no execution)."""
    hubbard = [h.model_dump() for h in req.hubbard]
    out: list[dict[str, Any]] = []
    for unit in req.units:
        try:
            if unit.calcType in _PW_CALCS:
                if req.structure is None:
                    raise HTTPException(
                        status_code=400,
                        detail=f"structure required for pw.x calc '{unit.calcType}' (unit {unit.id})",
                    )
                params = {**unit.params, "calculation": unit.calcType}
                text = generate_pw_input(
                    req.structure, params, prefix=req.prefix, ecutwfc=req.ecutwfc,
                    ecutrho=req.ecutrho, functional=req.functional, hubbard=hubbard,
                    outdir=unit.outdir,
                )
            else:
                text = generate_postproc_input(
                    unit.calcType, unit.params, prefix=req.prefix,
                    functional=req.functional, outdir=unit.outdir, name=unit.name,
                )
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"unit {unit.id}: {exc}") from exc
        out.append({
            "id": unit.id, "calcType": unit.calcType,
            "executable": _EXECUTABLE.get(unit.calcType), "input": text,
        })
    return GenerateResponse(units=out)


def _dft_io() -> Any:
    """Build the Firestore+GCS+Batch DftIO from settings (per request; tests monkeypatch)."""
    from src.config import get_settings
    from src.dft.io import FirestoreGcsBatchIO

    s = get_settings()
    return FirestoreGcsBatchIO(
        project=s.gcp_project_id, region=s.gcp_region, bucket=s.dft_bucket,
        image_uri=s.dft_image_uri, topic=s.dft_advance_topic,
        service_account=s.dft_batch_sa, machine_preset=s.dft_machine_preset,
        use_spot=s.dft_use_spot, max_run_sec=s.dft_max_run_sec, npool=s.dft_npool,
    )


@router.post("/submit")
def submit_workflow(req: SubmitRequest) -> dict[str, Any]:
    """Persist a DftWorkflow then launch its root units on Cloud Batch (one tick).

    Subsequent ticks are driven by Batch JobStateChanged notifications → /dft/advance.
    """
    wf = req.workflow
    if "structure" not in wf or "units" not in wf:
        raise HTTPException(status_code=400, detail="workflow needs 'structure' and 'units'")
    io = _dft_io()
    try:
        io.create_workflow(req.tenantId, req.workflowId, wf,
                           machine_preset=req.machinePreset, max_run_sec=req.maxRunSec,
                           npool=req.npool)
        overall = advance(io, req.tenantId, req.workflowId, None)
    except FatalError as exc:
        raise HTTPException(status_code=400, detail=str(exc)[:300]) from exc
    return {"workflowId": req.workflowId, "overallStatus": overall}


@router.post("/advance", status_code=status.HTTP_204_NO_CONTENT)
async def advance_workflow(request: Request) -> None:
    """Pub/Sub push endpoint for Batch job-state-change notifications.

    Batch puts JobUID/JobName/NewJobState/Type in message ATTRIBUTES (data is empty).
    We act only on terminal SUCCEEDED/FAILED, map JobName → labels → advance the DAG.

    Ack semantics (mirror the papers handler):
      204 → ack (success or ignored intermediate state) · 400 → ack, no retry (FatalError)
      5xx → nack, Pub/Sub retries → DLQ.
    """
    envelope: dict[str, Any] = await request.json()
    message = envelope.get("message")
    if not isinstance(message, dict):
        raise HTTPException(status_code=400, detail="Invalid Pub/Sub envelope")
    attrs = message.get("attributes") or {}
    if attrs.get("Type") != "JOB_STATE_CHANGED" or attrs.get("NewJobState") not in ("SUCCEEDED", "FAILED"):
        return  # 204 — ignore QUEUED/SCHEDULED/RUNNING/etc.
    job_name = attrs.get("JobName")
    if not job_name:
        return

    try:
        labels = get_job_labels(job_name)
        tenant_id = labels.get("dft_tenant")
        workflow_id = labels.get("dft_workflow")
        unit_id = labels.get("dft_unit")
        if not (tenant_id and workflow_id and unit_id):
            return  # not one of ours — ack
        advance(_dft_io(), tenant_id, workflow_id,
                {"unitId": unit_id, "state": attrs["NewJobState"]})
    except FatalError as exc:
        raise HTTPException(status_code=400, detail=str(exc)[:300]) from exc
    except Exception as exc:  # noqa: BLE001 — HTTP boundary; 5xx → Pub/Sub retry
        raise HTTPException(status_code=500, detail=str(exc)[:300]) from exc


# ── R272w-bands: band-structure plot data (on-demand, reads GCS .out) ──────────
class BandsRequest(BaseModel):
    tenantId: str
    workflowId: str


_GREEK_LABEL = {"GAMMA": "Γ", "DELTA": "Δ", "SIGMA": "Σ", "LAMBDA": "Λ"}


def _pmg_from_doc(struct: dict[str, Any]) -> Any:
    """Build a pymatgen Structure from the stored DftStructure (for the k-path)."""
    from pymatgen.core import Lattice, Structure

    cellp = struct.get("cellParameters")
    if cellp:
        latt = Lattice(cellp)
    else:
        from src.dft.structure_io import _cell_from_ibrav

        cd = {int(k): v for k, v in (struct.get("celldm") or {}).items()}
        cell = _cell_from_ibrav(int(struct.get("ibrav", 0)), cd)
        if cell is None:
            raise ValueError("cannot build cell: ibrav not in whitelist and no cellParameters")
        latt = Lattice(cell)
    pos = struct.get("atomicPositions") or []
    species = [p["element"] for p in pos]
    coords = [[p["x"], p["y"], p["z"]] for p in pos]
    return Structure(latt, species, coords)


@router.post("/bands")
def bands_plot(req: BandsRequest) -> dict[str, Any]:
    """Band-structure plot data for a completed workflow. No re-run.

    kdist = cumulative |Δk| (QE prints k in cartesian 2π/alat, so plain Euclidean);
    bands = per-band E−E_F over k; ticks = high-symmetry labels; plus Fermi + gap.
    """
    import math

    from src.dft.qe_parser import (
        band_gap_from_eigenvalues,
        parse_bands,
        parse_scf_summary,
    )

    io = _dft_io()
    doc = io.load(req.tenantId, req.workflowId)
    units = doc.get("units") or []
    by_calc = {u.get("calcType"): u for u in units}
    bands_u = by_calc.get("bands")
    scf_u = by_calc.get("scf") or by_calc.get("nscf")
    if not bands_u:
        raise HTTPException(status_code=404, detail="workflow has no 'bands' unit")

    bands_out = io.fetch_output(req.tenantId, req.workflowId, bands_u["id"])
    bres = parse_bands(bands_out)
    kp = bres.get("kpoints") or []
    ev = bres.get("eigenvalues") or []
    if not kp or not ev:
        raise HTTPException(status_code=422, detail="no eigenvalues parsed from bands output")

    fermi = None
    n_elec = None
    if scf_u:
        try:
            scf_out = io.fetch_output(req.tenantId, req.workflowId, scf_u["id"])
            summ = parse_scf_summary(scf_out)
            fermi = summ.get("fermi_ev")
            n_elec = summ.get("n_electrons")
        except Exception:  # noqa: BLE001 — Fermi/n_elec are optional enrichments
            pass

    shift = fermi or 0.0
    kdist = [0.0]
    for i in range(1, len(kp)):
        kdist.append(kdist[-1] + math.dist(kp[i], kp[i - 1]))
    kdist = [round(x, 6) for x in kdist]

    nbnd = len(ev[0])
    bands = [
        [round(ev[k][b] - shift, 4) for k in range(len(ev)) if b < len(ev[k])]
        for b in range(nbnd)
    ]

    ticks: list[dict[str, Any]] = []
    try:
        from src.dft.kpath import get_kpath

        pmg = _pmg_from_doc(doc.get("structure") or {})
        path = get_kpath(pmg).get("path") or []
        idx = 0
        for entry in path:
            if 0 <= idx < len(kdist):
                label = _GREEK_LABEL.get(entry["label"], entry["label"])
                if ticks and abs(ticks[-1]["dist"] - kdist[idx]) < 1e-6:
                    ticks[-1]["label"] = f"{ticks[-1]['label']}|{label}"
                else:
                    ticks.append({"dist": kdist[idx], "label": label})
            idx += int(entry.get("npoints", 1))
    except Exception:  # noqa: BLE001 — labels are best-effort; bands still render
        ticks = []

    gap = None
    if n_elec:
        try:
            gap = band_gap_from_eigenvalues(bres, n_elec)
        except Exception:  # noqa: BLE001
            gap = None

    return {
        "kdist": kdist,
        "bands": bands,
        "ticks": ticks,
        "nbnd": nbnd,
        "nk": len(kp),
        "fermiEv": fermi,
        "gap": gap,
    }
