"""Mistral OCR step.

Port labyra-app/src/lib/ai/rag/ocr/mistral.ts + pipeline/ocr-step.ts.

Flow:
  1. check_cancelled (Firestore poll)
  2. update_status('ocr')
  3. Download PDF từ Firebase Storage (reuse src/gcs_client)
  4. check_cancelled
  5. Mistral SDK: upload file → get signed URL → ocr.process
  6. Parse pages, compute cost
  7. Best-effort delete uploaded file (cleanup Mistral storage)
  8. check_cancelled
  9. increment_cost('ocr', cost_usd) + update pageCount field
  10. Return OcrResult Pydantic model

Idempotency: if paper.status already past 'ocr' (chunking/enriching/...),
caller orchestrator phải decide skip — ocr.py không tự skip. Reason: worker
có thể được trigger reprocess intentionally, ocr step không nên silently skip.

@phase R167-B2
"""
from __future__ import annotations

import logging
import time
from functools import lru_cache

from google.cloud import firestore  # type: ignore[import-untyped]
from mistralai.client.sdk import Mistral  # type: ignore[import-untyped]

from src.config import get_settings
from src.gcs_client import download_bytes
from src.papers.errors import FatalError, RetryableError
from src.papers.pricing import mistral_ocr_cost_usd
from src.papers.state import check_cancelled, increment_cost, update_status
from src.papers.types import OcrPage, OcrResult

logger = logging.getLogger(__name__)

MISTRAL_OCR_MODEL = "mistral-ocr-latest"


@lru_cache(maxsize=1)
def _mistral_client() -> Mistral:
    """Singleton Mistral client (lru_cache for worker instance reuse)."""
    settings = get_settings()
    if not settings.mistral_api_key:
        raise FatalError("MISTRAL_API_KEY missing in worker settings")
    return Mistral(api_key=settings.mistral_api_key)


def run_ocr_step(
    db: firestore.Client,
    tenant_id: str,
    paper_id: str,
    storage_path: str,
) -> OcrResult:
    """Run OCR step. Mirrors TS runOcrStep.

    Args:
        db: Firestore client
        tenant_id: tenant ID
        paper_id: paper ID
        storage_path: Firebase Storage path (e.g. 'papers/abc/file.pdf')

    Raises:
        CancelledError: user requested cancellation between steps
        FatalError: missing API key, malformed PDF, file not found
        RetryableError: Mistral API transient error (network, 5xx, timeout)
    """
    started_at = time.monotonic()

    check_cancelled(db, tenant_id, paper_id)
    update_status(db, tenant_id, paper_id, "ocr")

    # ── Download PDF từ GCS ─────────────────────────────────
    try:
        pdf_bytes = download_bytes(storage_path)
    except FileNotFoundError as exc:
        raise FatalError(f"PDF not found at {storage_path}") from exc
    except Exception as exc:
        # GCS transient (network, throttling) → retryable
        raise RetryableError(f"GCS download failed: {exc}") from exc

    if not pdf_bytes:
        raise FatalError(f"PDF empty at {storage_path}")

    logger.info(
        "ocr_download_done tenant=%s paper=%s bytes=%d",
        tenant_id, paper_id, len(pdf_bytes),
    )

    check_cancelled(db, tenant_id, paper_id)

    # ── Mistral OCR ─────────────────────────────────────────
    client = _mistral_client()
    file_id: str | None = None
    try:
        uploaded = client.files.upload(
            file={"file_name": "document.pdf", "content": pdf_bytes},
            purpose="ocr",
        )
        file_id = uploaded.id

        signed = client.files.get_signed_url(file_id=file_id)

        ocr_response = client.ocr.process(
            model=MISTRAL_OCR_MODEL,
            document={"type": "document_url", "document_url": signed.url},
        )
    except FatalError:
        raise
    except Exception as exc:  # Mistral SDK exceptions are not strongly typed
        # Treat as retryable — Mistral has transient 5xx, timeouts.
        # Fatal cases (invalid_pdf, malformed) caller orchestrator can downgrade.
        raise RetryableError(f"Mistral OCR failed: {exc}") from exc
    finally:
        # Best-effort cleanup (matches TS pattern)
        if file_id is not None:
            try:
                client.files.delete(file_id=file_id)
            except Exception as cleanup_exc:  # noqa: BLE001
                logger.warning("mistral_file_cleanup_failed file_id=%s err=%s", file_id, cleanup_exc)

    check_cancelled(db, tenant_id, paper_id)

    # ── Parse response ──────────────────────────────────────
    raw_pages = getattr(ocr_response, "pages", None) or []
    pages: list[OcrPage] = []
    for idx, p in enumerate(raw_pages):
        page_idx = getattr(p, "index", None)
        page_number = (page_idx + 1) if page_idx is not None else (idx + 1)
        markdown = getattr(p, "markdown", "") or ""
        pages.append(OcrPage(pageNumber=page_number, text=markdown))

    page_count = len(pages)
    full_text = "\n\n".join(p.text for p in pages)
    cost_usd = mistral_ocr_cost_usd(page_count)
    latency_ms = int((time.monotonic() - started_at) * 1000)

    # ── Cost + page count ───────────────────────────────────
    increment_cost(db, tenant_id, paper_id, "ocr", cost_usd)
    db.document(f"tenants/{tenant_id}/papers/{paper_id}").update({"pageCount": page_count})

    logger.info(
        "ocr_done tenant=%s paper=%s pages=%d cost_usd=%.6f latency_ms=%d",
        tenant_id, paper_id, page_count, cost_usd, latency_ms,
    )

    return OcrResult(
        fullText=full_text,
        pages=pages,
        pageCount=page_count,
        costUsd=cost_usd,
    )
