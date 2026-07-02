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

from src.config import get_settings
from src.dft import structure as dft_structure
from src.dft.generator import generate_postproc_input, generate_pw_input
from src.dft.scene import build_scene, export_structure
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


def _pseudo_bucket() -> Any:
    from google.cloud import storage

    s = get_settings()
    return storage.Client(project=s.gcp_project_id).bucket(s.dft_bucket)


class _PseudoUpload(_BaseModel):
    filename: str
    contentB64: str


@router.get("/pseudo/list")
def pseudo_list() -> dict[str, list[dict[str, Any]]]:
    """List the tenant's uploaded pseudopotential UPFs (GCS ``pseudo/`` prefix)."""
    try:
        bucket = _pseudo_bucket()
        out: list[dict[str, Any]] = []
        for blob in bucket.list_blobs(prefix=f"{_PSEUDO_PREFIX}/"):
            name = blob.name[len(_PSEUDO_PREFIX) + 1 :]
            if name and name.lower().endswith(".upf"):
                out.append({"filename": name, "element": _element_from_upf_name(name)})
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
        return {"filename": fn, "element": _element_from_upf_name(fn), "size": len(data)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"pseudo upload failed: {str(exc)[:200]}") from exc


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
        conv = parse_convergence(out_text)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422, detail=f"convergence parse failed: {str(exc)[:200]}"
        ) from exc
    conv["calcType"] = calc
    conv["unitId"] = unit["id"]
    conv["wallSeconds"] = parse_walltime(out_text)
    return conv
