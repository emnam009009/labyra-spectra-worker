"""Firestore state machine writers + readers cho paper pipeline.

Ports labyra-app/src/lib/ai/rag/pipeline/state.ts to Python.
Key difference: cancellation is POLL-BASED (check Firestore cancelRequestedAt
field) instead of signal-based (TS uses AbortSignal).

Reason: worker handles Pub/Sub push as discrete HTTP requests — no shared
process state cho AbortController. Polling Firestore mỗi step là source
of truth cross-process.

@phase R167-B1
"""
from __future__ import annotations

import logging
from typing import Any

from google.cloud import firestore  # type: ignore[import-untyped]
from google.cloud.firestore_v1 import Increment, SERVER_TIMESTAMP  # type: ignore[import-untyped]

from src.papers.errors import CancelledError, FatalError
from src.papers.types import PaperDoc, PaperStatus

logger = logging.getLogger(__name__)

# Cost field names — must match Firestore Paper.costUsd schema
CostField = str  # Literal["ocr", "enrichment", "embedding"]


def _paper_ref(db: firestore.Client, tenant_id: str, paper_id: str) -> firestore.DocumentReference:
    """Build Firestore document reference for paper."""
    return db.document(f"tenants/{tenant_id}/papers/{paper_id}")


# ----------------------------------------------------------------------------
# Status writers
# ----------------------------------------------------------------------------


def update_status(
    db: firestore.Client,
    tenant_id: str,
    paper_id: str,
    status: PaperStatus,
    extra_fields: dict[str, Any] | None = None,
) -> None:
    """Atomic status update with statusUpdatedAt timestamp.

    Mirrors TS updatePaperStatus(). Uses SERVER_TIMESTAMP for clock consistency
    across worker instances.
    """
    fields: dict[str, Any] = {
        "status": status,
        "statusUpdatedAt": SERVER_TIMESTAMP,
    }
    if extra_fields:
        fields.update(extra_fields)
    _paper_ref(db, tenant_id, paper_id).update(fields)
    logger.info("paper_status_update tenant=%s paper=%s status=%s", tenant_id, paper_id, status)


def set_error(
    db: firestore.Client,
    tenant_id: str,
    paper_id: str,
    error: str,
    is_retryable: bool = False,
) -> None:
    """Mark paper as errored.

    Mirrors TS setPaperError(). Trims error message to 500 chars (Firestore
    string indexing cost + UI display sanity).
    """
    trimmed = error[:500]
    if is_retryable:
        _paper_ref(db, tenant_id, paper_id).update({
            "error": trimmed,
            "retryCount": Increment(1),
            "statusUpdatedAt": SERVER_TIMESTAMP,
        })
        logger.warning(
            "paper_error_retryable tenant=%s paper=%s err=%s",
            tenant_id, paper_id, trimmed,
        )
    else:
        _paper_ref(db, tenant_id, paper_id).update({
            "status": "failed",
            "error": trimmed,
            "statusUpdatedAt": SERVER_TIMESTAMP,
            "processingCompletedAt": SERVER_TIMESTAMP,
        })
        logger.error(
            "paper_error_fatal tenant=%s paper=%s err=%s",
            tenant_id, paper_id, trimmed,
        )


def set_cancelled(db: firestore.Client, tenant_id: str, paper_id: str) -> None:
    """Mark paper as cancelled (user-initiated).

    Mirrors TS setPaperCancelled(). Sets terminal status; processingCompletedAt
    for audit trail.
    """
    _paper_ref(db, tenant_id, paper_id).update({
        "status": "cancelled",
        "statusUpdatedAt": SERVER_TIMESTAMP,
        "processingCompletedAt": SERVER_TIMESTAMP,
    })
    logger.info("paper_cancelled tenant=%s paper=%s", tenant_id, paper_id)


# ----------------------------------------------------------------------------
# Cost accounting
# ----------------------------------------------------------------------------


def increment_cost(
    db: firestore.Client,
    tenant_id: str,
    paper_id: str,
    cost_field: CostField,
    amount: float,
) -> None:
    """Atomically increment cost subfield + total.

    Mirrors TS incrementPaperCost(). Uses Firestore Increment for concurrent
    safety (chunks of same paper may have parallel embed/enrich calls).

    Args:
        cost_field: One of 'ocr', 'enrichment', 'embedding'
        amount: USD cost (must be >= 0). Negative would underflow total.

    Raises:
        FatalError: if amount < 0 (programming bug, not retryable)
    """
    if amount < 0:
        raise FatalError(f"Cost amount must be non-negative, got {amount}")
    if cost_field not in ("ocr", "enrichment", "embedding"):
        raise FatalError(f"Invalid cost_field: {cost_field}")

    _paper_ref(db, tenant_id, paper_id).update({
        f"costUsd.{cost_field}": Increment(amount),
        "costUsd.total": Increment(amount),
    })


# ----------------------------------------------------------------------------
# Cancellation polling (poll-based, NOT signal-based)
# ----------------------------------------------------------------------------


def check_cancelled(db: firestore.Client, tenant_id: str, paper_id: str) -> None:
    """Raise CancelledError nếu user đã request cancellation.

    Replaces TS throwIfCancelled(signal) — instead polls Firestore.

    Call this BEFORE every expensive step (OCR, embed, index, enrich). Cost:
    1 Firestore read per call (~0.000036 USD). Cheap insurance against
    wasting $X on OCR/embed for a cancelled job.

    Raises:
        CancelledError: if paper.cancelRequestedAt > 0 OR status in
                        ('cancelling', 'cancelled')
    """
    snap = _paper_ref(db, tenant_id, paper_id).get(["cancelRequestedAt", "status"])
    if not snap.exists:
        # Paper doc deleted mid-flight — treat as cancellation (don't continue
        # processing orphan). Fatal because no doc to update.
        raise FatalError(f"paper not found tenant={tenant_id} paper={paper_id}")

    data = snap.to_dict() or {}
    cancel_requested = int(data.get("cancelRequestedAt", 0) or 0)
    current_status = data.get("status", "")

    if cancel_requested > 0 or current_status in ("cancelling", "cancelled"):
        raise CancelledError()


# ----------------------------------------------------------------------------
# Read helpers
# ----------------------------------------------------------------------------


def load_paper(db: firestore.Client, tenant_id: str, paper_id: str) -> PaperDoc:
    """Load paper doc as Pydantic model.

    Raises:
        FatalError: if paper not found (no doc to process — cannot recover)
    """
    snap = _paper_ref(db, tenant_id, paper_id).get()
    if not snap.exists:
        raise FatalError(f"paper not found tenant={tenant_id} paper={paper_id}")
    data = snap.to_dict() or {}
    # Inject id (Firestore doesn't include doc id in to_dict)
    data["id"] = paper_id
    return PaperDoc.model_validate(data)
