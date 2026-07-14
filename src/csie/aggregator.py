"""
CSIE evidence aggregator (Step 1) + consistency checker (Step 2).

Security:
  - Every Firestore query scoped by tenantId (multi-tenant isolation)
  - Input validation: formula regex, ID length caps
  - DoS protection: max 20 measurements per run
  - PII-safe logging: hash IDs before logging

@phase R185-8a-csie-evidence-aggregation
"""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import UTC, datetime
from typing import Any

from src.csie.types import (
    ConsistencyCheck,
    CSIEResult,
    EvidenceItem,
    PhaseEvidence,
)

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

# DoS protection: cap measurements processed
MAX_MEASUREMENTS_PER_RUN = 20

# Min measurements to run CSIE — single spectrum is single-phase analysis only
MIN_MEASUREMENTS = 2

# Technique strength weights (how reliable each technique is for phase ID)
TECHNIQUE_STRENGTH: dict[str, float] = {
    "xrd": 1.00,        # Crystallographic ground truth
    "raman": 0.90,      # Strong vibrational fingerprint
    "ftir": 0.75,       # Functional groups, less polymorph-specific
    "pl": 0.70,         # Bandgap-specific but not unique
    "uvvis": 0.60,
    "uvvis_drs": 0.60,
    "tga": 0.40,        # Thermal events not phase-specific
}

# Match quality → numeric weight
MATCH_WEIGHT: dict[str, float] = {
    "excellent": 1.0,
    "good": 0.8,
    "fair": 0.5,
    "poor": 0.2,
    "missing": 0.0,
}

# Formula validation — same as app schema
FORMULA_RE = re.compile(r"^[A-Z][A-Za-z0-9()\[\]\-+.,·]{0,49}$")

# ID validation
ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")


# ── Utilities ─────────────────────────────────────────────────────────────────

def _hash_for_logs(value: str) -> str:
    """SHA-256 prefix for PII-safe logging."""
    return hashlib.sha256(value.encode()).hexdigest()[:12]


def _validate_inputs(tenant_id: str, sample_id: str) -> tuple[bool, str | None]:
    """Strict input validation. Returns (ok, error_msg)."""
    if not tenant_id or not ID_RE.match(tenant_id):
        return False, "invalid_tenant_id"
    if not sample_id or not ID_RE.match(sample_id):
        return False, "invalid_sample_id"
    return True, None


def _build_idempotency_key(tenant_id: str, sample_id: str, max_ts: int) -> str:
    """Idempotency key includes max measurement timestamp.

    Same input → same key → cached result reuse.
    """
    raw = f"{tenant_id}:{sample_id}:{max_ts}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


# ── Evidence aggregation (Step 1) ────────────────────────────────────────────

def _technique_strength(spectrum_type: str) -> float:
    """Map spectrum_type to strength weight."""
    return TECHNIQUE_STRENGTH.get(spectrum_type.lower(), 0.5)


def _extract_component_evidence(
    spectrum_id: str,
    spectrum_type: str,
    analysis_result: dict[str, Any] | None,
    target_formula: str,
) -> EvidenceItem | None:
    """Find evidence for `target_formula` in one analysis result."""
    if not analysis_result:
        return None

    deviation = analysis_result.get("deviationAnalysis") or {}
    mode = deviation.get("mode")

    notable: list[str] = []
    match_quality = "missing"
    intent_coverage = 0.0
    hypotheses_count = 0

    # Multi-phase path — look for component match
    if mode == "multi-phase":
        multi = deviation.get("multiPhase") or {}
        components = multi.get("components") or []
        per_comp_hyps = deviation.get("perComponentHypotheses") or {}

        for comp in components:
            if comp.get("formula") == target_formula:
                match_result = comp.get("match_result") or {}
                match_quality = match_result.get("quality_grade", "poor")
                intent_coverage = float(comp.get("intent_coverage", 0.0))
                hyps = per_comp_hyps.get(target_formula, [])
                hypotheses_count = len(hyps)
                # Surface highest-confidence rule names
                for h in hyps[:3]:
                    if isinstance(h, dict):
                        notable.append(h.get("name", "unknown rule"))
                break

    # Single-phase path
    elif mode == "single-phase":
        if deviation.get("referenceFormula") == target_formula:
            mr = deviation.get("matchResult") or {}
            match_quality = mr.get("quality_grade", "poor")
            intent_coverage = float(mr.get("match_rate", 0.0))
            hyps = deviation.get("hypotheses") or []
            hypotheses_count = len(hyps)
            for h in hyps[:3]:
                if isinstance(h, dict):
                    notable.append(h.get("name", "unknown rule"))

    return EvidenceItem(
        spectrum_id=spectrum_id,
        spectrum_type=spectrum_type,
        technique_strength=_technique_strength(spectrum_type),
        match_quality=match_quality,
        intent_coverage=round(intent_coverage, 3),
        hypotheses_count=hypotheses_count,
        notable_findings=notable[:3],
    )


def aggregate_evidence(
    declared_composition: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
) -> dict[str, PhaseEvidence]:
    """
    Aggregate per-phase evidence across measurements.

    Args:
        declared_composition: from Sample.composition (already-validated)
        measurements: list of { spectrumId, spectrumType, analysisResult }

    Returns:
        { formula: PhaseEvidence }
    """
    evidence_map: dict[str, PhaseEvidence] = {}

    for comp in declared_composition:
        formula = comp.get("formula", "").strip()
        role = comp.get("role", "matrix")

        if not formula or not FORMULA_RE.match(formula):
            continue

        evidence_map[formula] = PhaseEvidence(
            formula=formula,
            role=role,
            declared_in_sample=True,
        )

    for measurement in measurements:
        spectrum_id = measurement.get("spectrumId", "")
        spectrum_type = (measurement.get("spectrumType", "") or "").lower()
        analysis = measurement.get("analysisResult")

        if not spectrum_id or not spectrum_type:
            continue

        for formula, evidence in evidence_map.items():
            item = _extract_component_evidence(
                spectrum_id=spectrum_id,
                spectrum_type=spectrum_type,
                analysis_result=analysis,
                target_formula=formula,
            )
            if item is None:
                continue

            if item.match_quality == "missing" or item.intent_coverage < 0.3:
                evidence.spectra_missing.append(spectrum_id)
            else:
                evidence.spectra_supporting.append(item)

    return evidence_map


# ── Consistency check (Step 2) ───────────────────────────────────────────────

def _compute_consistency(evidence: PhaseEvidence, total_measurements: int) -> None:
    """Mutate evidence: assign verdict + score + reasoning. In-place."""
    supporting = evidence.spectra_supporting
    missing = evidence.spectra_missing

    if not supporting:
        evidence.verdict = "missing"
        evidence.consistency_score = 0.0
        evidence.reasoning.append(
            f"Declared in sample but not observed in any of "
            f"{total_measurements} spectra"
        )
        return

    # Weighted score across supporting evidence
    weight_sum = 0.0
    weighted_sum = 0.0
    for item in supporting:
        match_w = MATCH_WEIGHT.get(item.match_quality, 0.0)
        weight = item.technique_strength
        weighted_sum += match_w * item.intent_coverage * weight
        weight_sum += weight

    score = weighted_sum / weight_sum if weight_sum > 0 else 0.0
    evidence.consistency_score = round(score, 3)

    # Verdict
    n_supporting = len(supporting)
    n_missing = len(missing)

    if n_supporting >= 2 and score >= 0.6:
        evidence.verdict = "confirmed"
        evidence.reasoning.append(
            f"Confirmed across {n_supporting} techniques "
            f"({', '.join(sorted({e.spectrum_type for e in supporting}))})"
        )
    elif n_supporting == 1 and score >= 0.5:
        evidence.verdict = "partial"
        evidence.reasoning.append(
            f"Single-technique evidence only ({supporting[0].spectrum_type}). "
            f"Confirm via additional spectroscopy."
        )
    elif n_missing > n_supporting:
        evidence.verdict = "conflict"
        evidence.reasoning.append(
            f"Declared but missing from {n_missing}/{total_measurements} spectra. "
            f"Possible inconsistency with sample reality."
        )
    else:
        evidence.verdict = "partial"
        evidence.reasoning.append("Weak or inconsistent evidence across spectra.")


def run_csie(
    tenant_id: str,
    sample_id: str,
    declared_composition: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
) -> CSIEResult:
    """
    Top-level CSIE Step 1+2 dispatcher.

    Args:
        tenant_id: tenant scope (validated)
        sample_id: sample scope (validated)
        declared_composition: from Sample.composition
        measurements: list of analyzed spectra (capped at MAX_MEASUREMENTS_PER_RUN)

    Returns CSIEResult with consistency analysis.
    """
    # Validate
    ok, err = _validate_inputs(tenant_id, sample_id)
    if not ok:
        logger.warning("CSIE input rejected: %s", err)
        return CSIEResult(status="failed", notes=[err or "validation_failed"])

    # Insufficient data check
    if not measurements or len(measurements) < MIN_MEASUREMENTS:
        return CSIEResult(
            status="insufficient_data",
            notes=[f"Need at least {MIN_MEASUREMENTS} analyzed spectra; "
                   f"got {len(measurements)}"],
            idempotency_key="",
        )

    # DoS protection — cap measurements
    if len(measurements) > MAX_MEASUREMENTS_PER_RUN:
        logger.info(
            "CSIE truncating measurements: %d -> %d for tenant=%s sample=%s",
            len(measurements), MAX_MEASUREMENTS_PER_RUN,
            _hash_for_logs(tenant_id), _hash_for_logs(sample_id),
        )
        # Sort by analyzedAt desc, take freshest
        measurements = sorted(
            measurements,
            key=lambda m: m.get("analyzedAt", 0),
            reverse=True,
        )[:MAX_MEASUREMENTS_PER_RUN]

    if not declared_composition:
        return CSIEResult(
            status="insufficient_data",
            notes=["No composition declared; CSIE requires declared phases to "
                   "cross-validate. Use single-phase analysis instead."],
        )

    # Build idempotency key
    max_ts = max(int(m.get("analyzedAt", 0)) for m in measurements)
    idem_key = _build_idempotency_key(tenant_id, sample_id, max_ts)

    # Aggregate
    evidence_map = aggregate_evidence(declared_composition, measurements)

    # Consistency per phase
    n_measurements = len(measurements)
    for evidence in evidence_map.values():
        _compute_consistency(evidence, n_measurements)

    # Detect unexpected observations
    unexpected: list[str] = []
    for measurement in measurements:
        analysis = measurement.get("analysisResult") or {}
        deviation = analysis.get("deviationAnalysis") or {}
        if deviation.get("mode") == "multi-phase":
            multi = deviation.get("multiPhase") or {}
            unassigned = multi.get("unassigned_peaks") or []
            if unassigned and len(unassigned) >= 3:
                unexpected.append(
                    f"{len(unassigned)} unassigned peaks in "
                    f"{measurement.get('spectrumType', '?')} spectrum"
                )

    # Overall coherence = mean of declared phase scores
    if evidence_map:
        scores = [e.consistency_score for e in evidence_map.values()]
        coherence = round(sum(scores) / len(scores), 3)
    else:
        coherence = 0.0

    conflicts = sum(1 for e in evidence_map.values() if e.verdict == "conflict")
    spectrum_types = sorted({m.get("spectrumType", "?") for m in measurements})

    consistency = ConsistencyCheck(
        sample_id_hash=_hash_for_logs(sample_id),
        tenant_id_hash=_hash_for_logs(tenant_id),
        measurements_analyzed=n_measurements,
        spectrum_types_present=spectrum_types,
        declared_phases=list(evidence_map.values()),
        unexpected_observations=unexpected,
        overall_coherence_score=coherence,
        conflicts_count=conflicts,
    )

    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    logger.info(
        "CSIE ok tenant=%s sample=%s measurements=%d coherence=%.2f conflicts=%d",
        consistency.tenant_id_hash, consistency.sample_id_hash,
        n_measurements, coherence, conflicts,
    )

    return CSIEResult(
        status="ok",
        consistency=consistency,
        computed_at=now,
        idempotency_key=idem_key,
        notes=[],
    )
