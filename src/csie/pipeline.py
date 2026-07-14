"""
CSIE pipeline orchestrator.

Top-level entry point that:
  1. Validates inputs
  2. Checks rate limit
  3. Pulls Sample composition + measurements (tenant-scoped)
  4. Computes idempotency key, checks debounce
  5. Runs CSIE aggregator + consistency check
  6. Writes single-doc result

Non-blocking failures: any exception logged and returns failure status,
does NOT raise to caller (so spectrum pipeline keeps working).

@phase R185-8b-csie-integration
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from src.csie.aggregator import (
    MAX_MEASUREMENTS_PER_RUN,
    MIN_MEASUREMENTS,
    _build_idempotency_key,
    _hash_for_logs,
    run_csie,
)
from src.csie.ambiguity import handle_ambiguous
from src.csie.firestore_io import (
    check_rate_limit,
    fetch_analyzed_measurements,
    fetch_sample,
    should_skip_debounce,
    write_csie_result,
)
from src.csie.types import CSIEResult

logger = logging.getLogger(__name__)


def run_csie_for_sample(
    tenant_id: str,
    sample_id: str,
    *,
    force: bool = False,
) -> CSIEResult:
    """
    Top-level orchestrator. Safe to call concurrently; idempotent via debounce.

    Args:
        tenant_id: tenant scope
        sample_id: sample to analyze
        force: skip debounce check (manual refresh from UI)

    Returns:
        CSIEResult with status indicating outcome.
    """
    log_tid = _hash_for_logs(tenant_id)
    log_sid = _hash_for_logs(sample_id)

    # Rate limit
    if not check_rate_limit(tenant_id, max_per_hour=50):
        logger.info("CSIE rate limited: tenant=%s sample=%s", log_tid, log_sid)
        return CSIEResult(
            status="rate_limited",
            notes=["Per-tenant rate limit (50/hour) exceeded; try again later"],
        )

    # Fetch Sample
    sample = fetch_sample(tenant_id, sample_id)
    if not sample:
        return CSIEResult(
            status="failed",
            notes=["Sample not found or access denied"],
        )

    composition = sample.get("composition") or []
    if not composition:
        return CSIEResult(
            status="insufficient_data",
            notes=["Sample has no declared composition; CSIE requires "
                   "multi-phase declaration to cross-validate"],
        )

    # Fetch measurements
    measurements = fetch_analyzed_measurements(
        tenant_id, sample_id, limit=MAX_MEASUREMENTS_PER_RUN + 5,
    )
    if len(measurements) < MIN_MEASUREMENTS:
        return CSIEResult(
            status="insufficient_data",
            notes=[f"Need >= {MIN_MEASUREMENTS} analyzed spectra; "
                   f"got {len(measurements)}"],
        )

    # Compute idempotency key + debounce check
    max_ts = max(int(m.get("analyzedAt", 0)) for m in measurements)
    idem_key = _build_idempotency_key(tenant_id, sample_id, max_ts)

    if not force and should_skip_debounce(tenant_id, sample_id, idem_key):
        logger.info(
            "CSIE skipped (debounce): tenant=%s sample=%s key=%s",
            log_tid, log_sid, idem_key,
        )
        return CSIEResult(
            status="ok",
            notes=["Same input as last run within 5 min; result reused"],
            idempotency_key=idem_key,
            computed_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )

    # Run aggregator
    result = run_csie(
        tenant_id=tenant_id,
        sample_id=sample_id,
        declared_composition=composition,
        measurements=measurements,
    )

    # R185-9: detect ambiguous observations across hypotheses
    if result.status == "ok" and result.consistency is not None:
        all_hyps: list[dict[str, Any]] = []
        for m in measurements:
            dev = (m.get("analysisResult") or {}).get("deviationAnalysis") or {}
            for h in dev.get("hypotheses") or []:
                all_hyps.append(h)
            for h in dev.get("compositeHypotheses") or []:
                all_hyps.append(h)
            for hyps in (dev.get("perComponentHypotheses") or {}).values():
                if isinstance(hyps, list):
                    all_hyps.extend(hyps)

        try:
            ambiguous = handle_ambiguous(
                all_hypotheses=all_hyps,
                csie_consistency=result.consistency.to_dict(),
            )
            # Attach to consistency notes for storage
            if ambiguous:
                result.notes.append(
                    f"{len(ambiguous)} ambiguous observation(s) detected; "
                    f"discrimination experiments suggested"
                )
                # Embed into consistency dict for storage
                consistency_dict = result.consistency.to_dict()
                consistency_dict["ambiguous_observations"] = [a.to_dict() for a in ambiguous]
                # Hack: replace consistency to_dict with the augmented one
                result.consistency._ambiguous_dict = consistency_dict  # type: ignore[attr-defined]
        except Exception:
            logger.exception("Ambiguity handler failed (non-blocking)")

    # Persist
    if result.status == "ok":
        try:
            payload = result.to_dict()
            # Inject ambiguous observations into the consistency block if present
            if result.consistency is not None and hasattr(result.consistency, "_ambiguous_dict"):
                payload["consistency"] = result.consistency._ambiguous_dict  # type: ignore[attr-defined]
            write_csie_result(tenant_id, sample_id, payload)
        except Exception:
            logger.exception("CSIE write failed: tenant=%s sample=%s", log_tid, log_sid)
            result.notes.append("write_failed")

    return result
