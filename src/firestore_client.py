"""Firestore wrapper: read SpectrumMetadata, write AnalysisResult, status FSM."""

from __future__ import annotations

import logging
import time
from typing import Any

from google.cloud import firestore

from src.config import get_settings

logger = logging.getLogger(__name__)

# Status transitions (matches src/types/spectra.ts SpectrumStatus)
STATUS_UPLOADED = "uploaded"
STATUS_QUEUED = "queued"
STATUS_PROCESSING = "processing"
STATUS_ANALYZED = "analyzed"
STATUS_FAILED = "failed"


def _client() -> firestore.Client:
    settings = get_settings()
    return firestore.Client(project=settings.gcp_project_id)


def spectrum_ref(tenant_id: str, spectrum_id: str) -> firestore.DocumentReference:
    return _client().document(f"tenants/{tenant_id}/spectra/{spectrum_id}")


def analysis_ref(tenant_id: str, spectrum_id: str) -> firestore.DocumentReference:
    return _client().document(f"tenants/{tenant_id}/spectra/{spectrum_id}/analysis/latest")


def tenant_ref(tenant_id: str) -> firestore.DocumentReference:
    return _client().document(f"tenants/{tenant_id}")


def get_spectrum(tenant_id: str, spectrum_id: str) -> dict[str, Any] | None:
    snap = spectrum_ref(tenant_id, spectrum_id).get()
    if not snap.exists:
        return None
    return snap.to_dict()


def get_tenant_locale(tenant_id: str) -> str:
    """Tenant defaultLocale; falls back to settings.default_locale."""
    settings = get_settings()
    snap = tenant_ref(tenant_id).get()
    if not snap.exists:
        return settings.default_locale
    data = snap.to_dict() or {}
    return str(data.get("defaultLocale") or data.get("locale") or settings.default_locale)


def transition_status(
    tenant_id: str,
    spectrum_id: str,
    new_status: str,
    *,
    error_message: str | None = None,
) -> None:
    """Atomic status update with updatedAt timestamp."""
    update: dict[str, Any] = {
        "status": new_status,
        "updatedAt": int(time.time() * 1000),
    }
    if new_status == STATUS_ANALYZED:
        update["analyzedAt"] = int(time.time() * 1000)
        update["analysisVersion"] = get_settings().analysis_version
        update["errorMessage"] = firestore.DELETE_FIELD
    if error_message:
        update["errorMessage"] = error_message
    spectrum_ref(tenant_id, spectrum_id).update(update)
    logger.info("status[%s/%s] → %s", tenant_id, spectrum_id, new_status)


def write_analysis_result(
    tenant_id: str,
    spectrum_id: str,
    result: dict[str, Any],
) -> None:
    """Write AnalysisResult to /tenants/{tid}/spectra/{sid}/analysis/latest."""
    payload = {
        **result,
        "schemaVersion": 1,
        "analysisVersion": get_settings().analysis_version,
        "createdAt": int(time.time() * 1000),
    }
    analysis_ref(tenant_id, spectrum_id).set(payload)
    logger.info("AnalysisResult written: %s/%s", tenant_id, spectrum_id)


def write_quick_stats(tenant_id: str, spectrum_id: str, stats: dict[str, Any]) -> None:
    """Update SpectrumMetadata.quickStats with row count, x/y range, peak count."""
    spectrum_ref(tenant_id, spectrum_id).update({"quickStats": stats})

# ── R185-3a: Material profile fetch for deviation analysis ────────────────────

def _fetch_material_profile_uncached(formula: str) -> dict[str, Any] | None:
    """Uncached Firestore read of /materialProfiles/{formula}."""
    if not formula:
        return None
    db = _client()
    snap = db.collection("materialProfiles").document(formula).get()
    if not snap.exists:
        return None
    data = snap.to_dict()
    if data is None:
        return None
    data["id"] = snap.id
    return data


def get_material_profile(formula: str) -> dict[str, Any] | None:
    """Fetch /materialProfiles/{formula} with TTL LRU cache (R185-4f).

    Root collection (not tenant-scoped) — global scientific reference data.
    Cache hit ~95% in production for hot materials.
    """
    if not formula:
        return None
    from src.deviation.profile_cache import cached_get_material_profile
    return cached_get_material_profile(formula, _fetch_material_profile_uncached)

