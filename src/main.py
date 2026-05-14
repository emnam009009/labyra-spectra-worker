"""FastAPI entrypoint for Cloud Run. Receives Pub/Sub push messages."""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status

from src.ai.analyzer import analyze as ai_analyze
from src.config import get_settings
from src.firestore_client import (
    STATUS_ANALYZED,
    STATUS_FAILED,
    STATUS_PROCESSING,
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

    # 8. Combine + write
    combined = {
        "parsed": parsed,
        "ai": ai_result,
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
