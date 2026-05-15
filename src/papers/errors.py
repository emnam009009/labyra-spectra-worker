"""Paper pipeline exception types.

Retry semantics for Pub/Sub push handler:
  - CancelledError → 204 (ack, no retry — user cancelled)
  - FatalError → 400 (ack, no retry — permanent bad input)
  - RetryableError + everything else → 500 (nack, Pub/Sub retries → DLQ)

@phase R167-B1
"""
from __future__ import annotations


class PaperPipelineError(Exception):
    """Base class cho paper pipeline errors."""


class CancelledError(PaperPipelineError):
    """User requested cancellation via Firestore cancelRequestedAt flag.

    Pub/Sub handler returns 204 to ack — message removed, no retry.
    """

    def __init__(self) -> None:
        super().__init__("cancelled")


class FatalError(PaperPipelineError):
    """Permanent failure — should NOT be retried.

    Examples: unauthorized, quota_exceeded, invalid_pdf, malformed JSON,
    tenant/paper not found.

    Pub/Sub handler returns 400 → message acked, no retry.
    """


class RetryableError(PaperPipelineError):
    """Transient failure — Pub/Sub should retry.

    Examples: network timeout, vendor 503, Firestore contention.

    Pub/Sub handler returns 500 → message nacked, exponential backoff
    retry up to max-delivery-attempts (5), then to DLQ.
    """
