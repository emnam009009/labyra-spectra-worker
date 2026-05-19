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
import hashlib
import time
from functools import lru_cache

from google.cloud import firestore  # type: ignore[import-untyped]
from mistralai.client.sdk import Mistral  # type: ignore[import-untyped]

from src.config import get_settings
from src.gcs_client import blob_exists, download_bytes, upload_bytes
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
    # # R176-1e-mistral-timeout
    # 10-min timeout for large PDFs (~2000 pages at ~3 pages/sec)
    return Mistral(api_key=settings.mistral_api_key, timeout_ms=600_000)



# ─── R181: OCR cache @r181-applied ─────────────────────────────────
def _content_hash(pdf_bytes: bytes) -> str:
    """SHA256 of raw PDF bytes — used as cache key."""
    return hashlib.sha256(pdf_bytes).hexdigest()


def _cache_path(content_hash: str) -> str:
    """GCS relative path for OCR cache entry."""
    return f"ocr-cache/{content_hash}.json"


def _try_load_cache(content_hash: str) -> OcrResult | None:
    """Return cached OcrResult or None if miss. Errors → None (best-effort).

    @phase R181
    """
    path = _cache_path(content_hash)
    try:
        if not blob_exists(path):
            return None
        from src.gcs_client import download_text
        raw = download_text(path)
        result = OcrResult.model_validate_json(raw)
        logger.info("ocr_cache_hit hash=%s pages=%d", content_hash[:12], result.page_count)
        return result
    except Exception as exc:
        logger.warning("ocr_cache_read_failed hash=%s err=%s", content_hash[:12], exc)
        return None


def _save_cache(content_hash: str, result: OcrResult) -> None:
    """Persist OcrResult JSON to GCS. Errors logged but non-fatal."""
    path = _cache_path(content_hash)
    try:
        payload = result.model_dump_json(by_alias=True)
        upload_bytes(path, payload.encode("utf-8"), content_type="application/json")
        logger.info("ocr_cache_saved hash=%s pages=%d", content_hash[:12], result.page_count)
    except Exception as exc:
        logger.warning("ocr_cache_save_failed hash=%s err=%s", content_hash[:12], exc)


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
    # ── R181 OCR cache check (SHA256 content hash) @r181-applied ──────
    pdf_hash = _content_hash(pdf_bytes)
    cached = _try_load_cache(pdf_hash)
    if cached is not None:
        # Cache hit — skip Mistral, save ~$0.80/paper
        try:
            db.document(f"tenants/{tenant_id}/papers/{paper_id}").update({
                "pageCount": cached.page_count,
                "ocrCacheHit": True,
                "ocrContentHash": pdf_hash,
            })
        except Exception:
            pass
        elapsed_ms = int((time.monotonic() - started_at) * 1000)
        logger.info(
            "ocr_step_done tenant=%s paper=%s elapsed_ms=%d pages=%d cost_usd=0.0000 cache=hit",
            tenant_id, paper_id, elapsed_ms, cached.page_count,
        )
        return cached
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

    # R181: save to cache (best-effort, non-fatal)
    _save_cache(pdf_hash, OcrResult(
        fullText=full_text,
        pages=pages,
        pageCount=page_count,
        costUsd=cost_usd,
    ))
    try:
        db.document(f"tenants/{tenant_id}/papers/{paper_id}").update({
            "ocrCacheHit": False,
            "ocrContentHash": pdf_hash,
        })
    except Exception:
        pass
    return OcrResult(
        fullText=full_text,
        pages=pages,
        pageCount=page_count,
        costUsd=cost_usd,
    )
