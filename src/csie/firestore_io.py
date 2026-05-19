"""
Firestore I/O for CSIE.

Tenant isolation: every query/write includes tenantId in path.
Pulls Sample composition + analyzed Measurements; writes single-doc CSIE result.

@phase R185-8b-csie-integration
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Max measurements pulled per CSIE run (DoS protection at I/O layer too)
MAX_MEASUREMENTS_FETCH = 25  # slightly above aggregator cap to allow filtering


def fetch_sample(tenant_id: str, sample_id: str) -> dict[str, Any] | None:
    """Fetch Sample doc scoped to tenant. Returns None if not found or wrong tenant."""
    from src.firestore_client import _client

    db = _client()
    snap = (
        db.collection("tenants")
        .document(tenant_id)
        .collection("samples")
        .document(sample_id)
        .get()
    )
    if not snap.exists:
        return None
    data = snap.to_dict()
    if not data:
        return None
    # Defense in depth: verify tenant_id matches
    if data.get("tenantId") != tenant_id:
        logger.error(
            "Tenant mismatch: doc.tenantId=%s, query=%s — possible bug or tampering",
            data.get("tenantId"), tenant_id,
        )
        return None
    data["id"] = snap.id
    return data


def fetch_analyzed_measurements(
    tenant_id: str,
    sample_id: str,
    limit: int = MAX_MEASUREMENTS_FETCH,
) -> list[dict[str, Any]]:
    """
    Fetch analyzed spectra for given sample (tenant-scoped).

    Filters:
      - sampleId == sample_id
      - status == "analyzed"
    Ordered by analyzedAt desc, limited.
    """
    from src.firestore_client import _client

    db = _client()

    try:
        query = (
            db.collection("tenants")
            .document(tenant_id)
            .collection("spectra")
            .where("sampleId", "==", sample_id)
            .where("status", "==", "analyzed")
            .order_by("analyzedAt", direction="DESCENDING")
            .limit(limit)
        )
        snaps = list(query.stream())
    except Exception:
        logger.exception("Failed to query measurements")
        return []

    measurements: list[dict[str, Any]] = []
    for snap in snaps:
        meta = snap.to_dict() or {}
        # Defense: tenant check on each doc
        if meta.get("tenantId") != tenant_id:
            logger.warning("Skipping doc with mismatched tenantId: %s", snap.id)
            continue

        # Pull analysis result subdoc
        analysis_snap = (
            db.collection("tenants").document(tenant_id)
            .collection("spectra").document(snap.id)
            .collection("analysis").document("latest").get()
        )
        analysis = analysis_snap.to_dict() if analysis_snap.exists else None

        measurements.append({
            "spectrumId": snap.id,
            "spectrumType": meta.get("spectrumType"),
            "analyzedAt": meta.get("analyzedAt", 0),
            "analysisResult": analysis,
        })

    return measurements


def fetch_existing_csie(tenant_id: str, sample_id: str) -> dict[str, Any] | None:
    """Read latest CSIE result for sample."""
    from src.firestore_client import _client

    db = _client()
    snap = (
        db.collection("tenants").document(tenant_id)
        .collection("samples").document(sample_id)
        .collection("crossSpectrum").document("latest").get()
    )
    if not snap.exists:
        return None
    return snap.to_dict()


def write_csie_result(
    tenant_id: str,
    sample_id: str,
    result_dict: dict[str, Any],
) -> None:
    """Write CSIE result to latest doc. Overwrite mode."""
    from src.firestore_client import _client

    db = _client()
    (
        db.collection("tenants").document(tenant_id)
        .collection("samples").document(sample_id)
        .collection("crossSpectrum").document("latest")
        .set(result_dict)
    )


# ── Rate limit + debounce check ──────────────────────────────────────────────

def check_rate_limit(tenant_id: str, max_per_hour: int = 50) -> bool:
    """
    Per-tenant rate limit using Firestore _rate_limits.
    Returns True if allowed, False if rate limited.

    Schema: _rate_limits/{key} = { count, windowStartMs, expiresAt }
    Reuses pattern from R162 ADR-015.
    """
    from src.firestore_client import _client
    import time

    db = _client()
    key = f"csie:{tenant_id}"
    now_ms = int(time.time() * 1000)
    window_ms = 60 * 60 * 1000  # 1 hour

    ref = db.collection("_rate_limits").document(key)

    @firestore_transaction
    def _txn(tx: Any) -> bool:  # noqa: ANN401
        snap = ref.get(transaction=tx)
        data = snap.to_dict() if snap.exists else None

        if not data or now_ms - data.get("windowStartMs", 0) > window_ms:
            tx.set(ref, {
                "count": 1,
                "windowStartMs": now_ms,
                "expiresAt": now_ms + window_ms,
            })
            return True

        count = data.get("count", 0)
        if count >= max_per_hour:
            return False

        tx.update(ref, {"count": count + 1})
        return True

    try:
        return _txn()
    except Exception:
        logger.exception("Rate limit check failed; failing open for now")
        return True  # fail open to avoid blocking on transient errors


def firestore_transaction(fn):
    """Wrap function in Firestore transaction."""
    from src.firestore_client import _client
    db = _client()
    def wrapper():
        return db.run_transaction(lambda tx: fn(tx))
    return wrapper


# ── Debounce check ───────────────────────────────────────────────────────────

def should_skip_debounce(
    tenant_id: str,
    sample_id: str,
    new_idempotency_key: str,
    debounce_seconds: int = 300,
) -> bool:
    """
    Returns True if CSIE was recently computed with same idempotency key
    within debounce window — skip recomputation.
    """
    import time
    existing = fetch_existing_csie(tenant_id, sample_id)
    if not existing:
        return False

    existing_key = existing.get("idempotency_key", "")
    existing_at = existing.get("computed_at", "")

    if not existing_key or not existing_at:
        return False

    if existing_key != new_idempotency_key:
        return False  # input changed → must recompute

    # Same key + recent → skip
    try:
        from datetime import datetime, timezone
        existing_dt = datetime.fromisoformat(existing_at.replace("Z", "+00:00"))
        elapsed = (datetime.now(timezone.utc) - existing_dt).total_seconds()
        return elapsed < debounce_seconds
    except Exception:
        return False
