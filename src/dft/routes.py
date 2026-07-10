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

from fastapi import APIRouter, HTTPException, Request, status, Header

from src.config import get_settings
from src.dft import structure as dft_structure
from src.dft.generator import generate_postproc_input, generate_pw_input
from src.dft.scene import analyze_structure, brillouin_zone, build_scene, export_structure
from pydantic import BaseModel as _BaseModel
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
    "avgpot": "average.x",
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
        api_key = req.mp_api_key or get_settings().mp_api_key
        if not (req.mp_id and api_key):
            raise HTTPException(
                status_code=400,
                detail="mp_id required for source=mp_id (and an MP API key must be configured)",
            )
        return dft_structure.from_mp_id(
            req.mp_id, api_key, req.pseudo_map,
            use_primitive=req.use_primitive, prefer_ibrav=req.prefer_ibrav,
        )
    except HTTPException:
        raise
    except ValueError as exc:  # sanity failure / bad structure
        raise HTTPException(status_code=400, detail=str(exc)[:300]) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)[:300]) from exc


class _SceneRequest(_BaseModel):
    structure: dict[str, Any]


class _ExportRequest(_BaseModel):
    structure: dict[str, Any]
    fmt: str


@router.post("/structure/scene")
def structure_scene(req: _SceneRequest) -> dict[str, Any]:
    """Reconstruct a stored DftStructure → render scene (atoms + CrystalNN bonds)."""
    try:
        return build_scene(req.structure)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)[:300]) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)[:300]) from exc


@router.post("/structure/analysis")
def structure_analysis(req: _SceneRequest) -> dict[str, Any]:
    """Full crystallographic summary (symmetry, Wyckoff, density, oxidation, …)
    for the structure detail panel."""
    try:
        return analyze_structure(req.structure)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)[:300]) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)[:300]) from exc


@router.post("/structure/brillouin")
def structure_brillouin(req: _SceneRequest) -> dict[str, Any]:
    """First Brillouin zone (facets), high-symmetry k-points and band path for the
    reciprocal-space viewer."""
    try:
        return brillouin_zone(req.structure)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)[:300]) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)[:300]) from exc


@router.post("/structure/export")
def structure_export(req: _ExportRequest) -> dict[str, str]:
    """Emit CIF / POSCAR text for a stored DftStructure."""
    try:
        return {"fmt": req.fmt, "text": export_structure(req.structure, req.fmt)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)[:300]) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)[:300]) from exc


@router.post("/kpath")
def build_kpath(req: _SceneRequest) -> dict[str, Any]:
    """High-symmetry BZ path (seekpath) for a bands K_POINTS {crystal_b} input.

    Accepts a stored DftStructure (reconstructed to a pymatgen cell). seekpath
    standardizes the cell, so pair this path with the standardized primitive
    structure from /dft/structure (use_primitive=True).
    """
    from src.dft.kpath import get_kpath
    from src.dft.scene import _reconstruct

    try:
        st = _reconstruct(req.structure)
        return get_kpath(st)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"kpath failed: {str(exc)[:300]}") from exc


_PSEUDO_PREFIX = "pseudo"
_MAX_UPF_BYTES = 20 * 1024 * 1024  # 20 MB — generous for a single UPF


def _element_from_upf_name(name: str) -> str | None:
    """Best-effort element guess from a UPF filename (e.g. 'W.pbe-...UPF' → 'W',
    'Fe_ONCV...upf' → 'Fe'). Convention-based hint only; the user can reassign."""
    import re

    m = re.match(r"^([A-Z][a-z]?)(?=[._\-]|$)", name)
    return m.group(1) if m else None


_UPF_HEADER_BYTES = 131072  # PP_HEADER sits near the top; 128 KB covers PP_INFO


def _parse_upf_header(data: bytes) -> dict[str, Any]:
    """Author-suggested cutoffs + element from a UPF header (v2 XML attributes
    ``wfc_cutoff``/``rho_cutoff``; v1 'Suggested cutoff for wfc and rho' line).
    Values are in Ry, as written by the pseudopotential generator."""
    import re

    text = data.decode("utf-8", errors="ignore")
    out: dict[str, Any] = {}
    m = re.search(r"<PP_HEADER([^>]*)/?>", text, re.S)
    attrs = m.group(1) if m else ""
    for key, name in (("wfc_cutoff", "ecutwfc"), ("rho_cutoff", "ecutrho")):
        mm = re.search(rf'{key}\s*=\s*"([^"]+)"', attrs)
        if mm:
            try:
                val = float(mm.group(1))
            except ValueError:
                continue
            if val > 0:
                out[name] = val
    mz = re.search(r'z_valence\s*=\s*"\s*([\d.eE+-]+)\s*"', attrs)
    if mz:
        try:
            out["zValence"] = float(mz.group(1))
        except ValueError:
            pass
    if "zValence" not in out:
        mz1 = re.search(r"([\d.]+)\s+Z valence", text)
        if mz1:
            out["zValence"] = float(mz1.group(1))
    me = re.search(r'element\s*=\s*"\s*([A-Za-z]{1,2})\s*"', attrs)
    if me:
        out["element"] = me.group(1)
    # Pseudopotential kind → cutoff-ratio guidance (NC: ecutrho≈4·ecutwfc;
    # US/PAW: 8–12·ecutwfc). v2 has is_paw/is_ultrasoft/pseudo_type attributes;
    # v1 spells it in the header text ("Ultrasoft"/"Norm-conserving"/"Projector").
    ptype = None
    mt = re.search(r'pseudo_type\s*=\s*"\s*([A-Za-z]+)\s*"', attrs)
    if mt:
        raw = mt.group(1).upper()
        ptype = {"NC": "NC", "SL": "NC", "US": "US", "USPP": "US", "PAW": "PAW"}.get(raw, raw)
    if ptype is None:
        if re.search(r'is_paw\s*=\s*"\s*T', attrs):
            ptype = "PAW"
        elif re.search(r'is_ultrasoft\s*=\s*"\s*T', attrs):
            ptype = "US"
    if ptype is None:
        low = text[:4000].lower()
        if "paw" in low or "projector augmented" in low:
            ptype = "PAW"
        elif "ultrasoft" in low:
            ptype = "US"
        elif "norm-conserving" in low or "norm conserving" in low:
            ptype = "NC"
    if ptype:
        out["pseudoType"] = ptype
    # Relativistic treatment: scalar (SR) / full (FR) / none (NR).
    mr = re.search(r'relativistic\s*=\s*"\s*([A-Za-z]+)\s*"', attrs)
    if mr:
        out["relativistic"] = {
            "scalar": "SR",
            "full": "FR",
            "no": "NR",
            "none": "NR",
        }.get(mr.group(1).lower(), mr.group(1).upper())
    elif re.search(r"\bfull[ -]?relativistic\b", text[:4000], re.I) or re.search(r"\bfully[ -]?relativistic\b", text[:4000], re.I):
        out["relativistic"] = "FR"
    elif re.search(r"\bscalar[ -]?relativistic\b", text[:4000], re.I):
        out["relativistic"] = "SR"
    # Nonlinear core correction (semicore contribution to XC).
    mc = re.search(r'core_correction\s*=\s*"\s*([TF])', attrs)
    if mc:
        out["nlcc"] = mc.group(1) == "T"
    else:
        mc1 = re.search(r"nonlinear core[- ]correction\s*:?\s*(yes|no|t|f|true|false)", text[:4000], re.I)
        if mc1:
            out["nlcc"] = mc1.group(1).lower() in ("yes", "t", "true")
    # Functional label if present (e.g. PBE, PBESOL) — informational.
    mf = re.search(r'functional\s*=\s*"\s*([A-Za-z0-9 ]+?)\s*"', attrs)
    if mf:
        out["functional"] = mf.group(1).strip().split()[0].upper()
    if "ecutwfc" not in out:
        mv = re.search(r"([\d.]+)\s+([\d.]+)\s+Suggested cutoff for wfc and rho", text)
        if mv:
            out["ecutwfc"] = float(mv.group(1))
            out["ecutrho"] = float(mv.group(2))
    if "element" not in out:
        mel = re.search(r"^\s*([A-Z][a-z]?)\s+Element\s*$", text, re.M)
        if mel:
            out["element"] = mel.group(1)
    return out


def _pseudo_bucket() -> Any:
    from google.cloud import storage

    s = get_settings()
    return storage.Client(project=s.gcp_project_id).bucket(s.dft_bucket)


class _PseudoUpload(_BaseModel):
    filename: str
    contentB64: str


@router.get("/pseudo/list")
def pseudo_list() -> dict[str, list[dict[str, Any]]]:
    """List the tenant's uploaded pseudopotential UPFs (GCS ``pseudo/`` prefix)
    with author-suggested cutoffs parsed from each file's PP_HEADER."""
    try:
        bucket = _pseudo_bucket()
        out: list[dict[str, Any]] = []
        for blob in bucket.list_blobs(prefix=f"{_PSEUDO_PREFIX}/"):
            name = blob.name[len(_PSEUDO_PREFIX) + 1 :]
            if not name or not name.lower().endswith(".upf"):
                continue
            info: dict[str, Any] = {
                "filename": name,
                "element": _element_from_upf_name(name),
                "size": blob.size,
                "ecutwfc": None,
                "ecutrho": None,
                "pseudoType": None,
                "zValence": None,
                "relativistic": None,
                "nlcc": None,
                "functional": None,
            }
            try:
                head = blob.download_as_bytes(start=0, end=_UPF_HEADER_BYTES - 1)
                parsed = _parse_upf_header(head)
                info["ecutwfc"] = parsed.get("ecutwfc")
                info["ecutrho"] = parsed.get("ecutrho")
                info["pseudoType"] = parsed.get("pseudoType")
                info["zValence"] = parsed.get("zValence")
                info["relativistic"] = parsed.get("relativistic")
                info["nlcc"] = parsed.get("nlcc")
                info["functional"] = parsed.get("functional")
                if parsed.get("element"):
                    info["element"] = parsed["element"]
            except Exception:  # noqa: BLE001 — header parse is best-effort
                pass
            out.append(info)
        out.sort(key=lambda p: p["filename"].lower())
        return {"pseudos": out}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"pseudo list failed: {str(exc)[:200]}") from exc


@router.post("/pseudo/upload")
def pseudo_upload(req: _PseudoUpload) -> dict[str, Any]:
    """Upload one .UPF into the tenant's pseudopotential library (GCS ``pseudo/``).

    The entrypoint stages the whole ``pseudo/`` prefix into each run's pseudo_dir,
    so an uploaded UPF becomes available to pw.x once assigned in ATOMIC_SPECIES.
    """
    import base64
    import binascii

    fn = req.filename.strip().replace("\\", "/").split("/")[-1]
    if not fn or not fn.lower().endswith(".upf"):
        raise HTTPException(status_code=400, detail="expected a .UPF filename")
    try:
        data = base64.b64decode(req.contentB64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="invalid base64 content") from exc
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    if len(data) > _MAX_UPF_BYTES:
        raise HTTPException(status_code=413, detail="UPF exceeds 20 MB")
    try:
        blob = _pseudo_bucket().blob(f"{_PSEUDO_PREFIX}/{fn}")
        blob.upload_from_string(data, content_type="application/octet-stream")
        parsed = _parse_upf_header(data[:_UPF_HEADER_BYTES])
        return {
            "filename": fn,
            "element": parsed.get("element") or _element_from_upf_name(fn),
            "size": len(data),
            "ecutwfc": parsed.get("ecutwfc"),
            "ecutrho": parsed.get("ecutrho"),
            "pseudoType": parsed.get("pseudoType"),
            "zValence": parsed.get("zValence"),
            "relativistic": parsed.get("relativistic"),
            "nlcc": parsed.get("nlcc"),
            "functional": parsed.get("functional"),
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"pseudo upload failed: {str(exc)[:200]}") from exc


@router.post("/nbnd-suggest")
def nbnd_suggest(req: dict[str, Any]) -> dict[str, Any]:
    """Minimum nbnd from valence electrons for a structure + its assigned UPFs.

    n_electrons = Sum (z_valence of each atom's pseudopotential). For a spin-
    unpolarised insulator the occupied-band count is n_electrons/2; nbnd is then
    padded by +15% (a common headroom so a handful of empty bands are available
    for DOS tails and to avoid the highest band pinning the Fermi search). Spin-
    polarised runs carry bands in both channels. Returns the inputs so the UI can
    explain the number. Body: {structure, pseudoMap: {element: filename}, nspin?}.
    """
    import math

    structure = req.get("structure") or {}
    pseudo_map = req.get("pseudoMap") or {}
    nspin = int(req.get("nspin") or 1)
    positions = structure.get("atomicPositions") or []
    if not positions or not pseudo_map:
        raise HTTPException(status_code=422, detail="structure and pseudoMap required")

    bucket = _pseudo_bucket()
    z_by_el: dict[str, float] = {}
    for el, fn in pseudo_map.items():
        try:
            head = bucket.blob(f"{_PSEUDO_PREFIX}/{fn}").download_as_bytes(
                start=0, end=_UPF_HEADER_BYTES - 1
            )
            zv = _parse_upf_header(head).get("zValence")
            if zv:
                z_by_el[el] = float(zv)
        except Exception:  # noqa: BLE001
            pass

    counts: dict[str, int] = {}
    for a in positions:
        el = a.get("element")
        if el:
            counts[el] = counts.get(el, 0) + 1
    missing = sorted(set(counts) - set(z_by_el))
    if missing:
        raise HTTPException(
            status_code=422, detail=f"missing z_valence for: {', '.join(missing)}"
        )

    n_elec = sum(z_by_el[el] * n for el, n in counts.items())
    n_occ = n_elec if nspin == 2 else n_elec / 2.0
    nbnd = max(int(math.ceil(n_occ * 1.15)), int(math.ceil(n_occ)) + 4)
    return {
        "nElectrons": n_elec,
        "nOccupied": n_occ,
        "nbnd": nbnd,
        "headroomPct": 15,
        "nspin": nspin,
        "zValence": z_by_el,
    }


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
                    pseudo_map=req.pseudo_map, outdir=unit.outdir,
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
                           npool=req.npool, created_by=req.createdBy,
                           created_by_uid=req.createdByUid)
        overall = advance(io, req.tenantId, req.workflowId, None)
    except FatalError as exc:
        raise HTTPException(status_code=400, detail=str(exc)[:300]) from exc
    return {"workflowId": req.workflowId, "overallStatus": overall}


@router.post("/reconcile-sweep")
async def reconcile_sweep(x_cron_secret: str | None = Header(default=None)) -> dict[str, Any]:
    """Reconcile ALL running workflows (Cloud Scheduler entry point). Guarded by a
    shared secret in the ``X-Cron-Secret`` header vs ``DFT_CRON_SECRET`` — this
    endpoint mutates workflows across every tenant and has no user context, so it
    is refused unless the secret is configured and matches. No secret set → 503
    (feature disabled), rather than running unprotected."""
    from src.dft.driver import reconcile_all

    secret = get_settings().dft_cron_secret
    if not secret:
        raise HTTPException(status_code=503, detail="reconcile sweep disabled (no DFT_CRON_SECRET)")
    if x_cron_secret != secret:
        raise HTTPException(status_code=403, detail="bad or missing X-Cron-Secret")
    try:
        return reconcile_all(_dft_io())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"reconcile sweep failed: {str(exc)[:200]}") from exc


@router.post("/cancel")
async def cancel_workflow(request: Request) -> dict[str, Any]:
    """Stop a running workflow (or a single unit). Cancels the Batch job(s),
    releasing the VM, and marks the unit(s) failed with a 'cancelled by user'
    reason so the DAG stops. Body: {tenantId, workflowId, unitId?}."""
    from src.dft.driver import cancel

    body = await request.json()
    tenant_id = body.get("tenantId")
    workflow_id = body.get("workflowId")
    unit_id = body.get("unitId")
    if not tenant_id or not workflow_id:
        raise HTTPException(status_code=422, detail="tenantId and workflowId required")
    try:
        return cancel(_dft_io(), tenant_id, workflow_id, unit_id=unit_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"cancel failed: {str(exc)[:200]}") from exc


@router.post("/reconcile")
async def reconcile_workflow(request: Request) -> dict[str, Any]:
    """Actively poll Batch for a workflow's queued/running units and fail any that
    are stuck (unprovisionable → quota/capacity) or whose job has vanished. Safe to
    call repeatedly; intended for a client poll or a periodic Cloud Scheduler sweep.
    Body: {tenantId, workflowId}."""
    from src.dft.driver import reconcile

    body = await request.json()
    tenant_id = body.get("tenantId")
    workflow_id = body.get("workflowId")
    if not tenant_id or not workflow_id:
        raise HTTPException(status_code=422, detail="tenantId and workflowId required")
    try:
        return reconcile(_dft_io(), tenant_id, workflow_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"reconcile failed: {str(exc)[:200]}") from exc


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
            raise ValueError("cannot build cell: ibrav not whitelisted and no cellParameters")
        latt = Lattice(cell)
    pos = struct.get("atomicPositions") or []
    species = [p["element"] for p in pos]
    coords = [[p["x"], p["y"], p["z"]] for p in pos]
    return Structure(latt, species, coords)


@router.post("/bands")
async def bands_plot(request: Request) -> dict[str, Any]:
    """Band-structure plot data for a completed workflow. No re-run.

    kdist = cumulative |Δk| (QE prints k in cartesian 2π/alat → plain Euclidean);
    bands = per-band E−E_F over k; ticks = high-symmetry labels; plus Fermi + gap.
    """
    import math

    from src.dft.qe_parser import (
        band_gap_from_eigenvalues,
        parse_bands,
        parse_scf_summary,
    )

    body = await request.json()
    tenant_id = body.get("tenantId")
    workflow_id = body.get("workflowId")
    if not tenant_id or not workflow_id:
        raise HTTPException(status_code=422, detail="tenantId and workflowId required")

    io = _dft_io()
    doc = io.load(tenant_id, workflow_id)
    units = doc.get("units") or []
    by_calc = {u.get("calcType"): u for u in units}
    bands_u = by_calc.get("bands")
    scf_u = by_calc.get("scf") or by_calc.get("nscf")
    if not bands_u:
        raise HTTPException(status_code=404, detail="workflow has no 'bands' unit")

    bands_out = io.fetch_output(tenant_id, workflow_id, bands_u["id"])
    bres = parse_bands(bands_out)
    kp = bres.get("kpoints") or []
    ev = bres.get("eigenvalues") or []
    if not kp or not ev:
        raise HTTPException(status_code=422, detail="no eigenvalues parsed from bands output")

    fermi = None
    n_elec = None
    if scf_u:
        try:
            scf_out = io.fetch_output(tenant_id, workflow_id, scf_u["id"])
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
    except Exception:  # noqa: BLE001 — labels best-effort; bands still render
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


_RY_EV = 13.605693122994
_BOHR_A = 0.529177210903


@router.post("/avgpot")
async def avgpot_plot(request: Request) -> dict[str, Any]:
    """Planar/macroscopic-averaged electrostatic potential V(z) from an
    ``avgpot`` unit's avg.dat (average.x: columns z[bohr], planar[Ry], macro[Ry]).
    Returned converted to Å / eV. vacuumEv = max of the macroscopic (or planar)
    curve — the vacuum plateau for a slab with enough vacuum."""
    body = await request.json()
    tenant_id = body.get("tenantId")
    workflow_id = body.get("workflowId")
    unit_id = body.get("unitId")
    if not tenant_id or not workflow_id:
        raise HTTPException(status_code=422, detail="tenantId and workflowId required")

    io = _dft_io()
    doc = io.load(tenant_id, workflow_id)
    units = doc.get("units") or []
    unit = next(
        (u for u in units if u.get("id") == unit_id) if unit_id
        else (u for u in units if u.get("calcType") == "avgpot"),
        None,
    )
    if not unit or unit.get("calcType") != "avgpot":
        raise HTTPException(status_code=404, detail="workflow has no 'avgpot' unit")

    try:
        names = io.list_blobs(f"workflows/{workflow_id}/units/{unit['id']}/")
        dat = [n for n in names if n.endswith("avg.dat")]
        if not dat:
            raise HTTPException(status_code=404, detail="avg.dat not found (unit finished?)")
        z: list[float] = []
        planar: list[float] = []
        macro: list[float] = []
        for line in io.read_text(dat[0]).splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                vals = [float(x) for x in parts[:3]]
            except ValueError:
                continue
            z.append(vals[0] * _BOHR_A)
            planar.append(vals[1] * _RY_EV)
            macro.append((vals[2] if len(vals) > 2 else vals[1]) * _RY_EV)
        if not z:
            raise HTTPException(status_code=422, detail="avg.dat parsed to zero rows")
        ref = macro if any(abs(v) > 1e-12 for v in macro) else planar
        return {
            "unitId": unit["id"],
            "z": z,
            "planar": planar,
            "macro": macro,
            "vacuumEv": max(ref),
            "nPoints": len(z),
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422, detail=f"avgpot read/parse failed: {str(exc)[:200]}"
        ) from exc


@router.post("/dos")
async def dos_plot(request: Request) -> dict[str, Any]:
    """Total DOS (dos.x fildos) + element/orbital-projected DOS (projwfc filpdos)
    for a completed workflow. No re-run. Data files are discovered by listing the
    unit dir (robust to the functional/prefix-derived filenames)."""
    from src.dft.qe_parser import parse_dos, parse_pdos, parse_scf_summary

    body = await request.json()
    tenant_id = body.get("tenantId")
    workflow_id = body.get("workflowId")
    if not tenant_id or not workflow_id:
        raise HTTPException(status_code=422, detail="tenantId and workflowId required")

    io = _dft_io()
    doc = io.load(tenant_id, workflow_id)
    units = doc.get("units") or []
    by_calc = {u.get("calcType"): u for u in units}
    dos_u = by_calc.get("dos")
    pdos_u = by_calc.get("pdos")
    if not dos_u and not pdos_u:
        raise HTTPException(status_code=404, detail="workflow has no 'dos' or 'pdos' unit")

    energies = None
    total = None
    fermi = None
    if dos_u:
        try:
            names = io.list_blobs(f"workflows/{workflow_id}/units/{dos_u['id']}/")
            dos_files = [n for n in names if n.endswith(".dos")]
            if dos_files:
                d = parse_dos(io.read_text(dos_files[0]))
                energies = d.get("energies_ev")
                total = d.get("dos")
                fermi = d.get("fermi_ev")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=422, detail=f"DOS read/parse failed: {str(exc)[:200]}"
            ) from exc

    pdos_series: list[dict[str, Any]] = []
    if pdos_u:
        try:
            names = io.list_blobs(f"workflows/{workflow_id}/units/{pdos_u['id']}/")
            pdos_names = [n for n in names if "pdos_atm#" in n]
            files = {n.split("/")[-1]: io.read_text(n) for n in pdos_names}
            p = parse_pdos(files)
            pdos_series = p.get("pdos") or []
            if energies is None:
                energies = p.get("energies_ev")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=422, detail=f"PDOS read/parse failed: {str(exc)[:200]}"
            ) from exc

    if fermi is None:
        scf_u = by_calc.get("scf") or by_calc.get("nscf")
        if scf_u:
            try:
                summ = parse_scf_summary(io.fetch_output(tenant_id, workflow_id, scf_u["id"]))
                fermi = summ.get("fermi_ev")
            except Exception:  # noqa: BLE001
                pass

    return {
        "energies": energies or [],
        "total": total,
        "pdos": pdos_series,
        "fermiEv": fermi,
        "nPoints": len(energies or []),
    }


@router.post("/results")
async def results_summary(request: Request) -> dict[str, Any]:
    """Consolidated scientific summary: bands gap, DOS@Fermi, PDOS character at
    VBM/CBM, spin/magnetization, total energy, electrons. No re-run."""
    from src.dft.qe_parser import (
        band_gap_from_eigenvalues,
        parse_bands,
        parse_dos,
        parse_pdos,
        parse_scf_summary,
        pdos_character,
    )

    body = await request.json()
    tenant_id = body.get("tenantId")
    workflow_id = body.get("workflowId")
    if not tenant_id or not workflow_id:
        raise HTTPException(status_code=422, detail="tenantId and workflowId required")

    io = _dft_io()
    doc = io.load(tenant_id, workflow_id)
    units = doc.get("units") or []
    by_calc = {u.get("calcType"): u for u in units}
    res: dict[str, Any] = {}

    # scf/nscf: energy, fermi, electrons, spin
    n_elec = None
    spin_pol = False
    scf_u = by_calc.get("scf") or by_calc.get("nscf")
    if scf_u:
        try:
            summ = parse_scf_summary(io.fetch_output(tenant_id, workflow_id, scf_u["id"]))
            n_elec = summ.get("n_electrons")
            spin_pol = bool(summ.get("spin_polarized"))
            res["totalEnergyRy"] = summ.get("total_energy_ry")
            res["fermiEv"] = summ.get("fermi_ev")
            res["nElectrons"] = n_elec
            res["scfIterations"] = summ.get("scf_iterations")
            res["spin"] = {
                "spinPolarized": spin_pol,
                "totalMag": summ.get("total_mag"),
                "absMag": summ.get("abs_mag"),
            }
            if summ.get("homo_ev") is not None:
                res["scfGap"] = {
                    "homoEv": summ.get("homo_ev"),
                    "lumoEv": summ.get("lumo_ev"),
                    "gapEv": summ.get("band_gap_ev"),
                }
        except Exception:  # noqa: BLE001
            pass

    # bands: k-resolved gap
    gap = None
    bands_u = by_calc.get("bands")
    if bands_u and n_elec:
        try:
            bres = parse_bands(io.fetch_output(tenant_id, workflow_id, bands_u["id"]))
            gap = band_gap_from_eigenvalues(bres, n_elec, spin_polarized=spin_pol)
            res["bandGap"] = gap
        except Exception:  # noqa: BLE001
            pass

    # dos: DOS at Fermi
    energies = None
    dos_u = by_calc.get("dos")
    if dos_u:
        try:
            names = io.list_blobs(f"workflows/{workflow_id}/units/{dos_u['id']}/")
            dos_files = [n for n in names if n.endswith(".dos")]
            if dos_files:
                d = parse_dos(io.read_text(dos_files[0]))
                energies = d.get("energies_ev")
                res["dos"] = {
                    "fermiEv": d.get("fermi_ev"),
                    "dosAtFermi": d.get("dos_at_fermi"),
                    "nPoints": d.get("n_points"),
                }
        except Exception:  # noqa: BLE001
            pass

    # pdos: orbital character at VBM/CBM
    pdos_u = by_calc.get("pdos")
    if pdos_u and gap:
        try:
            names = io.list_blobs(f"workflows/{workflow_id}/units/{pdos_u['id']}/")
            pn = [n for n in names if "pdos_atm#" in n]
            files = {n.split("/")[-1]: io.read_text(n) for n in pn}
            p = parse_pdos(files)
            pe = p.get("energies_ev")
            ps = p.get("pdos") or []
            res["pdosCharacter"] = {
                "vbm": pdos_character(pe, ps, gap["vbm_ev"]),
                "cbm": pdos_character(pe, ps, gap["cbm_ev"]),
            }
        except Exception:  # noqa: BLE001
            pass

    return res


@router.post("/convergence")
async def convergence_history(request: Request) -> dict[str, Any]:
    """SCF accuracy + per-ionic-step energy/force for a relax/vc-relax/scf unit.
    Picks vc-relax > relax > scf by default, or an explicit unitId. No re-run."""
    from src.dft.qe_parser import parse_convergence, parse_walltime

    body = await request.json()
    tenant_id = body.get("tenantId")
    workflow_id = body.get("workflowId")
    if not tenant_id or not workflow_id:
        raise HTTPException(status_code=422, detail="tenantId and workflowId required")

    io = _dft_io()
    doc = io.load(tenant_id, workflow_id)
    units = doc.get("units") or []
    by_calc = {u.get("calcType"): u for u in units}

    unit = None
    calc = None
    req_uid = body.get("unitId")
    if req_uid:
        unit = next((u for u in units if u.get("id") == req_uid), None)
        calc = unit.get("calcType") if unit else None
    if not unit:
        for ct in ("vc-relax", "relax", "scf"):
            if by_calc.get(ct):
                unit = by_calc[ct]
                calc = ct
                break
    if not unit:
        raise HTTPException(status_code=404, detail="no vc-relax/relax/scf unit")

    try:
        out_text = io.fetch_output(tenant_id, workflow_id, unit["id"])
    except Exception as exc:  # noqa: BLE001
        # The .out is not in GCS yet — the job is still QUEUED (no VM), or hasn't
        # reached its first 30 s streaming flush. That is not an error: return an
        # empty, not-done convergence so the UI shows "waiting for output" and the
        # client keeps polling.
        msg = str(exc).lower()
        if "no such object" in msg or "not found" in msg or "404" in msg:
            return {
                "calcType": calc,
                "unitId": unit["id"],
                "pending": True,
                "job_done": False,
                "converged": False,
                "scf_accuracy": [],
                "scf_seconds": [],
                "ionic_steps": [],
                "final_scf_accuracy": None,
                "walltimeSec": None,
            }
        raise HTTPException(
            status_code=502, detail=f"convergence fetch failed: {str(exc)[:200]}"
        ) from exc

    try:
        conv = parse_convergence(out_text)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422, detail=f"convergence parse failed: {str(exc)[:200]}"
        ) from exc
    conv["calcType"] = calc
    conv["unitId"] = unit["id"]
    conv["wallSeconds"] = parse_walltime(out_text)
    return conv
