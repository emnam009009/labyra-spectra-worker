"""DFT pipeline exception types — mirror src/papers/errors retry semantics.

Pub/Sub push handler maps:
  CancelledError → 204 (ack, user cancelled — no retry)
  FatalError → 400 (ack, permanent bad input — no retry)
  RetryableError + everything else → 500 (nack → Pub/Sub retries → DLQ)

@phase R272w-h (DFT P1-3)
"""
from __future__ import annotations


class DftPipelineError(Exception):
    """Base class for DFT pipeline errors."""


class CancelledError(DftPipelineError):
    def __init__(self) -> None:
        super().__init__("cancelled")


class FatalError(DftPipelineError):
    """Permanent failure — not retried (bad payload, workflow not found, invalid DAG)."""


class RetryableError(DftPipelineError):
    """Transient failure — Pub/Sub retries (Firestore contention, GCS/Batch 5xx)."""
