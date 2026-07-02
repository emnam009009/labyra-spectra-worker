"""FastAPI entrypoint for Cloud Run. Receives Pub/Sub push messages."""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status

from src.ai.analyzer import analyze as ai_analyze
from src.deviation.pipeline import run_deviation_analysis
from src.config import get_settings
from src.firestore_client import (
    STATUS_ANALYZED,
    STATUS_FAILED,
    STATUS_PROCESSING,
    get_material_profile,
    get_spectrum,
    get_tenant_locale,
    transition_status,
    write_analysis_result,
    write_quick_stats,
)
from src.gcs_client import download_bytes, download_text
from src.parsers import get_parser

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

app = FastAPI(title="labyra-spectra-worker", version="0.2.0")


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "labyra-spectra-worker", "version": "0.2.0"}


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Cloud Run health probe (separate from root for monitoring)."""
    return {"status": "ok"}


@app.post("/pubsub", status_code=status.HTTP_204_NO_CONTENT)
async def handle_pubsub_push(request: Request) -> None:
    """Pub/Sub push subscription endpoint.

    Returns 204 on success → Pub/Sub acks.
    Returns 4xx → Pub/Sub does NOT retry (permanent failure).
    Returns 5xx → Pub/Sub retries with exponential backoff up to DLQ.
    """
    envelope: dict[str, Any] = await request.json()

    if "message" not in envelope:
        raise HTTPException(status_code=400, detail="Invalid Pub/Sub envelope")

    message = envelope["message"]
    encoded = message.get("data", "")
    if not encoded:
        logger.warning("Empty message data, acking")
        return

    try:
        payload: dict[str, Any] = json.loads(base64.b64decode(encoded).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        logger.exception("Decode failed")
        # 400 → no retry (permanent: bad payload)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # R164-phase-5a: accept both legacy spectrumId + new measurementId (same UUID)
    # New app (R164+) sends measurementId; legacy app (R160-R163) sends spectrumId.
    # Internal variable stays `spectrum_id` to avoid touching downstream parsers.
    tenant_id = payload.get("tenantId")
    spectrum_id = payload.get("measurementId") or payload.get("spectrumId")
    # Optional `collection` field — switches Firestore path (default "spectra").
    # Phase 5b app will send "measurements"; legacy sends nothing → falls back.
    collection = payload.get("collection", "spectra")
    if not tenant_id or not spectrum_id:
        raise HTTPException(status_code=400, detail="Missing tenantId or measurementId/spectrumId")
    logger.info(
        "Decoded payload: tenant=%s id=%s collection=%s",
        tenant_id, spectrum_id, collection,
    )

    try:
        _process(tenant_id, spectrum_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Processing failed for %s/%s", tenant_id, spectrum_id)
        try:
            transition_status(tenant_id, spectrum_id, STATUS_FAILED, error_message=str(exc)[:500])
        except Exception:  # noqa: BLE001
            logger.exception("Failed to write failed-status")
        # 500 → Pub/Sub retries (transient assumption)
        raise HTTPException(status_code=500, detail=str(exc)[:200]) from exc


def _process(tenant_id: str, spectrum_id: str) -> None:
    """Core pipeline. Raises on error; caller writes failed status + acks."""
    settings = get_settings()

    # 1. Load metadata
    metadata = get_spectrum(tenant_id, spectrum_id)
    if not metadata:
        # Spectrum was deleted before processing — ack (don't retry)
        logger.warning("Spectrum not found, skipping: %s/%s", tenant_id, spectrum_id)
        return

    current_status = metadata.get("status")
    if current_status == STATUS_ANALYZED:
        logger.info("Already analyzed, skipping: %s/%s", tenant_id, spectrum_id)
        return

    # 2. Mark processing
    transition_status(tenant_id, spectrum_id, STATUS_PROCESSING)

    # 3. Dispatch parser
    spectrum_type = metadata["spectrumType"]
    sample_label = metadata.get("sampleLabel") or metadata.get("sample_label")
    chemical_formula = metadata.get("chemicalFormula") or metadata.get("chemical_formula")
    anode = metadata.get("anode")  # X-ray anode: Cu/Mo/Co/Cr/Fe/Ag, default Cu
    cv_n_electrons = int(metadata.get("nElectrons") or metadata.get("n_electrons") or 1)
    cv_scan_rate = metadata.get("scanRate") or metadata.get("scan_rate_v_s")
    cv_area = metadata.get("electrodeArea") or metadata.get("area_cm2")
    lsv_area = metadata.get("electrodeArea") or metadata.get("area_cm2")
    lsv_ref = metadata.get("referenceElectrode") or metadata.get("reference")
    lsv_ph = metadata.get("pH") or metadata.get("ph")
    lsv_reaction = metadata.get("reaction")  # her | oer
    lsv_ir = bool(metadata.get("irCorrected") or metadata.get("ir_corrected"))
    eis_area = metadata.get("electrodeArea") or metadata.get("area_cm2")
    eis_n = int(metadata.get("nElectrons") or metadata.get("n_electrons") or 1)
    eis_temp = float(metadata.get("temperatureK") or metadata.get("temperature_k") or 298.15)
    eis_format = metadata.get("dataFormat") or metadata.get("data_format")
    pec_area = metadata.get("electrodeArea") or metadata.get("area_cm2")
    pec_light_power = metadata.get("lightPower") or metadata.get("light_power_mw_cm2")
    pec_bias = metadata.get("appliedBias") or metadata.get("applied_bias_v")
    pec_eps_r = metadata.get("dielectricConstant") or metadata.get("eps_r")
    pec_ms_temp = float(
        metadata.get("temperatureK") or metadata.get("temperature_k") or 298.15
    )
    pec_ms_freqs = metadata.get("frequenciesHz") or metadata.get("frequencies_hz")
    ftir_mode = metadata.get("samplingMode") or metadata.get("mode")  # FTIR: transmission | atr
    dsc_heating_rate = metadata.get("heatingRate") or metadata.get("heating_rate")
    dsc_sample_mass = metadata.get("sampleMass") or metadata.get("sample_mass")
    dsc_polymer = metadata.get("polymer")
    dsc_y_unit = metadata.get("yUnit") or metadata.get("heatFlowUnit")
    laser_wl_meta = metadata.get("laserWavelength") or metadata.get("laser_wavelength")  # Raman excitation (nm)
    monochromator = metadata.get("monochromator")  # none/ni_filter/graphite/ge111/johansson/si220
    profile_function = metadata.get("profileFunction") or "pseudo_voigt"  # R161-phase-E
    zero_shift = float(metadata.get("zeroShift") or 0.0)  # 2θ correction in degrees
    
    # XRD: use citation-enabled parser (lookup COD + MP candidates)
    # Also support .xlsx via bytes-aware wrapper
    if spectrum_type == "xrd":
        from src.parsers.xrd import parse_xrd_with_citation, parse_xrd_bytes
        original_filename = metadata.get("originalFilename", "")
        if original_filename.lower().endswith(".xlsx"):
            def parser(raw_bytes_or_text):
                # If raw is bytes, parse xlsx; else fall through to text
                raw_bytes = raw_bytes_or_text if isinstance(raw_bytes_or_text, bytes) else raw_bytes_or_text.encode("utf-8")
                parsed = parse_xrd_bytes(
                    raw_bytes, original_filename,
                    anode=anode,
                    monochromator=monochromator,
                    profile=profile_function,
                    zero_shift=zero_shift,
                )
                # Wire hkl from top candidate into peaks (matches parse_xrd_with_citation behavior)
                from src.citation.lookup import lookup_xrd_candidates
                if parsed.get("peaks"):
                    citation_result = lookup_xrd_candidates(
                        parsed["peaks"],
                        sample_label=sample_label,
                        chemical_formula=chemical_formula,
                        filename=original_filename,
                    )
                    parsed["citation"] = citation_result
                    candidates = citation_result.get("candidates", [])
                    if candidates:
                        top = candidates[0]
                        user_hkl_map = top.get("user_hkl_map", {})
                        for user_idx, hkl in user_hkl_map.items():
                            idx = int(user_idx) if isinstance(user_idx, str) else user_idx
                            if 0 <= idx < len(parsed["peaks"]) and hkl:
                                parsed["peaks"][idx]["hkl"] = " ".join(str(int(v)) for v in hkl)
                return parsed
        else:
            parser = lambda raw: parse_xrd_with_citation(
                raw,
                sample_label=sample_label,
                chemical_formula=chemical_formula,
                filename=original_filename,
                anode=anode,
                monochromator=monochromator,
                profile=profile_function,
                zero_shift=zero_shift,
            )
    elif spectrum_type == "raman":
        from src.parsers.raman import parse_raman
        _laser = float(laser_wl_meta) if laser_wl_meta else None
        parser = lambda raw: parse_raman(raw, laser_wavelength=_laser)
    elif spectrum_type == "dsc":
        from src.parsers.dsc import parse_dsc
        _hr = float(dsc_heating_rate) if dsc_heating_rate else None
        _mass = float(dsc_sample_mass) if dsc_sample_mass else None
        parser = lambda raw: parse_dsc(
            raw, heating_rate_c_min=_hr, sample_mass_mg=_mass,
            polymer=dsc_polymer, y_unit=dsc_y_unit,
        )
    elif spectrum_type == "ftir":
        from src.parsers.ftir import parse_ftir
        parser = lambda raw: parse_ftir(raw, mode=ftir_mode)
    elif spectrum_type == "eis":
        from src.parsers.eis import parse_eis
        _area = float(eis_area) if eis_area else None
        parser = lambda raw: parse_eis(
            raw, area_cm2=_area, n_electrons=eis_n,
            temperature_k=eis_temp, data_format=eis_format,
        )
    elif spectrum_type == "lsv":
        from src.parsers.lsv import parse_lsv
        _a = float(lsv_area) if lsv_area else None
        _ph = float(lsv_ph) if lsv_ph is not None else None
        parser = lambda raw: parse_lsv(
            raw, area_cm2=_a, reference=lsv_ref, ph=_ph,
            reaction=lsv_reaction, ir_corrected=lsv_ir,
        )
    elif spectrum_type == "cv":
        from src.parsers.cv import parse_cv
        _sr = float(cv_scan_rate) if cv_scan_rate else None
        _a = float(cv_area) if cv_area else None
        parser = lambda raw: parse_cv(
            raw, n_electrons=cv_n_electrons, scan_rate_v_s=_sr, area_cm2=_a,
        )
    elif spectrum_type == "tafel":
        from src.parsers.tafel import parse_tafel
        _a = float(lsv_area) if lsv_area else None
        _ph = float(lsv_ph) if lsv_ph is not None else None
        parser = lambda raw: parse_tafel(
            raw, reference=lsv_ref, ph=_ph, area_cm2=_a, reaction=lsv_reaction,
        )
    elif spectrum_type == "pec_jv":
        from src.parsers.pec_jv import parse_pec_jv
        _a = float(pec_area) if pec_area else None
        _lp = float(pec_light_power) if pec_light_power else None
        _bias = float(pec_bias) if pec_bias else None
        parser = lambda raw: parse_pec_jv(
            raw, area_cm2=_a, light_power_mw_cm2=_lp, applied_bias_v=_bias,
        )
    elif spectrum_type == "pec_mott_schottky":
        from src.parsers.pec_mott_schottky import parse_pec_mott_schottky
        _a = float(pec_area) if pec_area else None
        _eps = float(pec_eps_r) if pec_eps_r else None
        _ph = float(lsv_ph) if lsv_ph is not None else None
        parser = lambda raw: parse_pec_mott_schottky(
            raw, eps_r=_eps, reference=lsv_ref, ph=_ph, area_cm2=_a,
            temperature_k=pec_ms_temp, frequencies_hz=pec_ms_freqs,
        )
    else:
        parser = get_parser(spectrum_type)

    # 4. Download raw bytes
    gs_url = metadata["storage"]["raw"]
    # For xlsx (binary), use download_bytes; else text
    original_filename_for_dl = metadata.get("originalFilename", "")
    if original_filename_for_dl.lower().endswith(".xlsx"):
        raw_text = download_bytes(gs_url)  # bytes for openpyxl
    else:
        raw_text = download_text(gs_url)

    # 5. Parse
    parsed = parser(raw_text)
    logger.info(
        "Parsed %s: peaks=%d, rows=%d",
        spectrum_type,
        len(parsed.get("peaks", [])),
        parsed.get("quick_stats", {}).get("rowCount", 0),
    )

    # 6. Persist quick stats
    if quick_stats := parsed.get("quick_stats"):
        write_quick_stats(tenant_id, spectrum_id, quick_stats)

    # 7. AI analyze
    locale = get_tenant_locale(tenant_id)
    ai_result = ai_analyze(parsed, metadata, locale)

    # 7b. R185-3a + R185-4: Deviation analysis (non-blocking)
    # Routes to multi-phase when Sample.composition declared, else single-phase fallback.
    deviation_result = None
    composition = metadata.get("composition")  # list of {formula, role, nominalFraction}
    if chemical_formula or composition:
        try:
            laser_wl = laser_wl_meta
            material_profile = get_material_profile(chemical_formula) if chemical_formula else None

            deviation_result = run_deviation_analysis(
                spectrum_type=spectrum_type,
                parsed=parsed,
                material_profile=material_profile,
                laser_wavelength=laser_wl,
                composition=composition,
                profile_loader=get_material_profile,
            )
            if deviation_result:
                mode = deviation_result.get("mode")
                if mode == "multi-phase":
                    mp = deviation_result["multiPhase"]
                    logger.info(
                        "Deviation (multi-phase): components=%d match_rate=%.0f%% grade=%s missing=%s",
                        len(mp["components"]),
                        mp["overall_match_rate"] * 100,
                        mp["overall_grade"],
                        mp["intended_but_not_observed"],
                    )
                elif mode == "single-phase":
                    mr = deviation_result["matchResult"]
                    logger.info(
                        "Deviation (single-phase): formula=%s match_rate=%.0f%% grade=%s hypotheses=%d",
                        chemical_formula,
                        mr["match_rate"] * 100,
                        mr["quality_grade"],
                        len(deviation_result["hypotheses"]),
                    )
        except Exception:  # noqa: BLE001
            logger.exception("Deviation analysis failed (non-blocking)")
            deviation_result = None

    # 8. Combine + write
    combined = {
        "parsed": parsed,
        "ai": ai_result,
        "deviationAnalysis": deviation_result,
        "locale": locale,
        "spectrumType": spectrum_type,
    }
    write_analysis_result(tenant_id, spectrum_id, combined)

    # 9. Mark analyzed
    transition_status(tenant_id, spectrum_id, STATUS_ANALYZED)
    logger.info("Done: %s/%s", tenant_id, spectrum_id)

    # GCS lifecycle policy handles raw file deletion after 7 days
    # (configured at bucket level, not per-object)



# ============================================================
# R160-spectra-4a-pdf: Reference card parser endpoint
# ============================================================
from pydantic import BaseModel
from src.reference.parser import parse_reference_card


class ParseReferenceCardRequest(BaseModel):
    text: str


# ============================================================================
# R167-A: Papers processing skeleton
# ----------------------------------------------------------------------------
# Async pipeline trigger via Pub/Sub push from topic 'paper-processing'.
# Skeleton (R167-A): acks message + updates Firestore status='received'.
# Full pipeline (OCR → chunk → embed → index → citations) ships in R167-B.
#
# Message shape (ADR-018):
#   {
#     "jobId": "uuid",
#     "tenantId": "tenant-dev-001",
#     "paperId": "abc123",
#     "version": 2,
#     "storagePath": "papers/abc123/file.pdf",
#     "createdBy": "uid",
#     "enqueuedAt": 1234567890
#   }
# ============================================================================


@app.get("/papers/health")
def papers_health() -> dict[str, str]:
    """Health check riêng cho papers pipeline."""
    return {
        "status": "ok",
        "subsystem": "papers",
        "phase": "R167-B6",
    }


@app.post("/papers/process", status_code=status.HTTP_204_NO_CONTENT)
async def handle_papers_push(request: Request) -> None:
    """Pub/Sub push subscription endpoint cho paper processing jobs.

    Auth: enforced bởi Cloud Run IAM (--push-auth-service-account).
    Worker không cần verify OIDC token trong code — Cloud Run reject 401 nếu
    Pub/Sub không attach valid token.

    Ack semantics (mirror /pubsub measurement handler):
      - 204 → ack, Pub/Sub không retry
      - 400 → ack, Pub/Sub KHÔNG retry (permanent: bad payload)
      - 5xx → nack, Pub/Sub retry với exponential backoff → DLQ sau 5 attempts

    R167-A scope: parse + validate + Firestore status update only.
    Pipeline call (OCR/embed/index/citations) thêm trong R167-B.
    """
    envelope: dict[str, Any] = await request.json()
    if "message" not in envelope:
        raise HTTPException(status_code=400, detail="Invalid Pub/Sub envelope")

    message = envelope["message"]
    encoded = message.get("data", "")
    if not encoded:
        logger.warning("papers: empty message data, acking")
        return

    try:
        payload: dict[str, Any] = json.loads(base64.b64decode(encoded).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        logger.exception("papers: decode failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Validate required fields per ADR-018 message shape
    job_id = payload.get("jobId")
    tenant_id = payload.get("tenantId")
    paper_id = payload.get("paperId")
    version = payload.get("version")
    storage_path = payload.get("storagePath")

    missing = [
        name for name, val in [
            ("jobId", job_id),
            ("tenantId", tenant_id),
            ("paperId", paper_id),
            ("version", version),
            ("storagePath", storage_path),
        ] if val is None
    ]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required fields: {', '.join(missing)}",
        )

    message_id = message.get("messageId", "unknown")
    logger.info(
        "papers: job=%s tenant=%s paper=%s v=%s msgId=%s",
        job_id, tenant_id, paper_id, version, message_id,
    )

    # R167-B6: wire orchestrator. Ack semantics:
    #   - success           → 204 (Pub/Sub ack)
    #   - CancelledError    → 204 (user cancelled, ack — no point retrying)
    #   - FatalError        → 400 (permanent failure, ack — no retry)
    #   - RetryableError +  → 500 (transient, Pub/Sub retries → DLQ after 5)
    #     any other Exception
    from src.papers.orchestrator import process_paper
    from src.papers.errors import CancelledError as _CancelledError, FatalError as _FatalError
    from google.cloud import firestore as _firestore

    created_by = payload.get("createdBy", "")
    db = _firestore.Client(project=get_settings().gcp_project_id)

    try:
        process_paper(
            db=db,
            tenant_id=tenant_id,
            paper_id=paper_id,
            storage_path=storage_path,
            job_id=job_id,
            created_by=created_by,
        )
    except _CancelledError:
        logger.info("papers: cancelled job=%s paper=%s — acking", job_id, paper_id)
        return  # 204
    except _FatalError as exc:
        # 400 = permanent → Pub/Sub does NOT retry
        raise HTTPException(status_code=400, detail=str(exc)[:200]) from exc
    except Exception as exc:  # noqa: BLE001 — final catch for HTTP boundary
        # 5xx → Pub/Sub retries up to max-delivery-attempts (5) → DLQ
        logger.exception("papers: pipeline failed job=%s paper=%s", job_id, paper_id)
        raise HTTPException(status_code=500, detail=str(exc)[:200]) from exc


@app.post("/reference/parse")
async def parse_reference(req: ParseReferenceCardRequest) -> dict:
    """Parse user-pasted XRD reference card text into structured peaks.

    Stateless: no auth, no Firestore. Just text → structured data.
    Persistence handled by app's /api/reference-cards endpoint.
    """
    try:
        result = parse_reference_card(req.text)
        return {"success": True, "data": result}
    except ValueError as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Reference parse failed")
        return {"success": False, "error": f"unexpected: {exc}"}

# ============================================================================
# R184: Materials Project sync endpoint
# ============================================================================

from pydantic import BaseModel as _BaseModel
from typing import Optional as _Optional


class MaterialSyncRequest(_BaseModel):
    formulas: list[str] | None = None  # None = sync DEFAULT_FORMULAS


class MaterialSearchRequest(_BaseModel):
    query: str
    limit: _Optional[int] = 30


@app.post("/materials/sync")
async def sync_materials(req: MaterialSyncRequest, request: Request) -> dict:
    """
    Trigger Materials Project API sync for given formulas.
    Updates /materialProfiles/{formula} in Firestore (merge).

    Auth: Bearer token required (superadmin enforced by caller convention;
    worker trusts Cloud Run IAM for internal calls).

    POST /materials/sync
    Body: {"formulas": ["MoS2", "WO3"]}  // omit for full DEFAULT_FORMULAS batch
    """
    from src.materials.mp_sync import sync_batch, DEFAULT_FORMULAS
    from google.cloud import firestore as _fs

    settings = get_settings()
    if not settings.mp_api_key:
        return {"error": "MP_API_KEY not configured", "status": "error"}

    formulas = req.formulas or DEFAULT_FORMULAS
    db = _fs.Client(project=settings.gcp_project_id)

    results = sync_batch(formulas, settings.mp_api_key, db)

    ok = [r for r in results if r["status"] == "ok"]
    not_found = [r for r in results if r["status"] == "not_found"]
    errors = [r for r in results if r["status"] == "error"]

    return {
        "total": len(formulas),
        "ok": len(ok),
        "not_found": len(not_found),
        "errors": len(errors),
        "results": results,
    }

# ── R185-4f: cache observability ──────────────────────────────────────────────

@app.post("/materials/search")
async def search_materials_endpoint(req: MaterialSearchRequest, request: Request) -> dict:
    """
    Search Materials Project by mp-id / chemical system / elements / formula and
    return a lean result list for the structure-import picker (no Firestore write).

    POST /materials/search
    Body: {"query": "WO3", "limit": 30}
    """
    from src.materials.mp_sync import search_materials

    settings = get_settings()
    if not settings.mp_api_key:
        return {"error": "MP_API_KEY not configured", "results": []}

    q = (req.query or "").strip()
    if not q:
        return {"query": "", "count": 0, "results": []}

    limit = max(1, min(req.limit or 30, 100))
    results = search_materials(q, settings.mp_api_key, limit)
    return {"query": q, "count": len(results), "results": results}


@app.get("/materials/cache-stats")
def materials_cache_stats() -> dict:
    """Return materialProfile cache hit/miss stats. Ops monitoring."""
    from src.deviation.profile_cache import cache_stats
    return cache_stats()


@app.post("/materials/cache-clear", status_code=status.HTTP_204_NO_CONTENT)
async def materials_cache_clear(request: Request) -> None:
    """Clear materialProfile cache. Call after admin updates profiles."""
    from src.deviation.profile_cache import cache_clear
    cache_clear()


# ── R185-8b: CSIE endpoints ──────────────────────────────────────────────────

@app.post("/csie/{sample_id}/refresh", status_code=status.HTTP_200_OK)
async def csie_refresh(sample_id: str, request: Request) -> dict:
    """
    Manual CSIE refresh trigger from UI.

    Requires OIDC auth + tenantId in request body (validated by Cloud Run IAM).
    Force=True skips debounce.
    """
    from src.csie.pipeline import run_csie_for_sample

    try:
        body = await request.json()
    except Exception:
        return {"status": "failed", "notes": ["invalid_json_body"]}

    tenant_id = body.get("tenantId")
    if not tenant_id:
        return {"status": "failed", "notes": ["missing_tenant_id"]}

    force = bool(body.get("force", False))
    result = run_csie_for_sample(tenant_id, sample_id, force=force)
    return result.to_dict()


@app.post("/csie/process", status_code=status.HTTP_200_OK)
async def csie_process(request: Request) -> dict:
    """
    Pub/Sub-triggered CSIE compute endpoint.

    Subscriber URL: receives push from csie-trigger topic with body
    { tenantId, sampleId } (base64 wrapped in Pub/Sub envelope).
    """
    from src.csie.pipeline import run_csie_for_sample
    import base64
    import json

    try:
        envelope = await request.json()
    except Exception:
        return {"status": "failed", "notes": ["invalid_envelope"]}

    # Pub/Sub envelope: { message: { data: base64, ... } }
    message = envelope.get("message") or {}
    raw_data = message.get("data")
    if not raw_data:
        return {"status": "failed", "notes": ["missing_message_data"]}

    try:
        decoded = base64.b64decode(raw_data).decode()
        payload = json.loads(decoded)
    except Exception:
        return {"status": "failed", "notes": ["malformed_message"]}

    tenant_id = payload.get("tenantId")
    sample_id = payload.get("sampleId")
    if not tenant_id or not sample_id:
        return {"status": "failed", "notes": ["missing_fields"]}

    result = run_csie_for_sample(tenant_id, sample_id, force=False)
    return result.to_dict()



# ── R272w-c: DFT input-generator endpoints (Phase 0) ──
from src.dft.routes import router as dft_router  # noqa: E402

app.include_router(dft_router)


_DFT_POSTPROC = {"ppbands", "dos", "pdos", "charge", "avgpot"}


@app.post("/dft/preview")
async def dft_preview(request: Request) -> dict[str, str]:
    """Render the QE .in for one unit from current params — no save, no run.

    Body: {calcType, structure, global, params}. Returns {"input": "<.in text>"}.
    Lets the UI double-check the exact input before launching (a 1-char QE error
    fails the whole job). Mirrors FirestoreGcsBatchIO._render.
    """
    from src.dft.generator import generate_postproc_input, generate_pw_input

    body = await request.json()
    calc = body.get("calcType") or body.get("calc")
    if not isinstance(calc, str):
        raise HTTPException(status_code=422, detail="calcType required")
    structure = body.get("structure") or {}
    g = body.get("global") or {}
    params = body.get("params") or {}
    _kp = params.get("kPoints")
    if isinstance(_kp, dict) and "shift" not in _kp:
        _kp["shift"] = [0, 0, 0]
    prefix = g.get("prefix") or "preview"
    functional = g.get("functional", "pbe")
    try:
        if calc in _DFT_POSTPROC:
            text = generate_postproc_input(
                calc,
                params,
                prefix=prefix,
                functional=functional,
                outdir="./out",
                name=params.get("name"),
            )
        else:
            ecutwfc = float(g.get("ecutwfc", 50.0))
            ecutrho = float(g.get("ecutrho", ecutwfc * 4.0))
            text = generate_pw_input(
                structure,
                params,
                prefix=prefix,
                ecutwfc=ecutwfc,
                ecutrho=ecutrho,
                functional=functional,
                hubbard=g.get("hubbard"),
                pseudo_map=g.get("pseudoMap"),
                outdir="./out",
            )
    except Exception as exc:  # preview is read-only — surface any render error as 422
        raise HTTPException(status_code=422, detail=f"render failed: {exc}") from exc
    return {"input": text}
