"""OCR step — engine-agnostic (Mistral / Datalab Marker).

Port labyra-app/src/lib/ai/rag/ocr + pipeline/ocr-step.ts.

Engine chosen by settings.ocr_engine ("mistral" default | "datalab");
settings.ocr_fallback ("mistral") runs if the primary engine fails. The OCR cache,
cost accounting, cancellation checks and status updates wrap the engine call and are
engine-agnostic.

@phase R167-B2, R221 (engine-agnostic + Datalab Marker)
"""
from __future__ import annotations

import contextlib
import hashlib
import logging
import time
from functools import lru_cache

from google.cloud import firestore  # type: ignore[import-untyped]
from mistralai.client.sdk import Mistral  # type: ignore[import-untyped]

from src.config import get_settings
from src.gcs_client import blob_exists, download_bytes, upload_bytes
from src.papers.errors import FatalError, RetryableError
from src.papers.ocr_datalab import datalab_ocr
from src.papers.pricing import datalab_ocr_cost_usd, mistral_ocr_cost_usd
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
    # R176-1e-mistral-timeout: 10-min timeout for large PDFs (~2000 pages @ ~3 p/s)
    return Mistral(api_key=settings.mistral_api_key, timeout_ms=600_000)


# ─── R181: OCR cache @r181-applied ─────────────────────────────────
def _content_hash(pdf_bytes: bytes) -> str:
    """SHA256 of raw PDF bytes — used as cache key."""
    return hashlib.sha256(pdf_bytes).hexdigest()


def _cache_path(content_hash: str) -> str:
    """GCS relative path for OCR cache entry."""
    return f"ocr-cache/{content_hash}.json"


def _try_load_cache(content_hash: str) -> OcrResult | None:
    """Return cached OcrResult or None if miss. Errors -> None (best-effort)."""
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


def _mistral_ocr(pdf_bytes: bytes) -> list[OcrPage]:
    """Mistral OCR engine: upload -> signed URL -> ocr.process -> parse pages."""
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
        raise RetryableError(f"Mistral OCR failed: {exc}") from exc
    finally:
        # Best-effort cleanup of uploaded file (Mistral storage)
        if file_id is not None:
            try:
                client.files.delete(file_id=file_id)
            except Exception as cleanup_exc:
                logger.warning(
                    "mistral_file_cleanup_failed file_id=%s err=%s", file_id, cleanup_exc
                )

    raw_pages = getattr(ocr_response, "pages", None) or []
    pages: list[OcrPage] = []
    for idx, page in enumerate(raw_pages):
        page_idx = getattr(page, "index", None)
        page_number = (page_idx + 1) if page_idx is not None else (idx + 1)
        markdown = getattr(page, "markdown", "") or ""
        pages.append(OcrPage(pageNumber=page_number, text=markdown))
    return pages


def _run_ocr_engine(pdf_bytes: bytes) -> tuple[list[OcrPage], str]:
    """Dispatch to the configured OCR engine with optional fallback.

    Returns (pages, engine_used). FatalError (e.g. bad key) is never silently
    swallowed; only transient failures fall back when ocr_fallback is set.
    """
    settings = get_settings()
    engine = (settings.ocr_engine or "mistral").strip().lower()

    if engine == "datalab":
        try:
            return datalab_ocr(pdf_bytes), "datalab"
        except FatalError:
            raise
        except Exception as exc:
            fallback = (settings.ocr_fallback or "").strip().lower()
            if fallback == "mistral":
                logger.error("ocr_datalab_failed_fallback_mistral err=%s", exc)
                return _mistral_ocr(pdf_bytes), "mistral"
            raise RetryableError(f"Datalab OCR failed (no fallback): {exc}") from exc

    return _mistral_ocr(pdf_bytes), "mistral"


def _engine_cost_usd(engine: str, page_count: int) -> float:
    """Cost for the engine that actually produced the result."""
    if engine == "datalab":
        return datalab_ocr_cost_usd(page_count)
    return mistral_ocr_cost_usd(page_count)


def run_ocr_step(
    db: firestore.Client,
    tenant_id: str,
    paper_id: str,
    storage_path: str,
) -> OcrResult:
    """Run OCR step. Mirrors TS runOcrStep (engine-agnostic since R221).

    Args:
        db: Firestore client
        tenant_id: tenant ID
        paper_id: paper ID
        storage_path: Firebase Storage path (e.g. 'papers/abc/file.pdf')

    Raises:
        CancelledError: user requested cancellation between steps
        FatalError: missing API key, malformed PDF, file not found
        RetryableError: OCR engine transient error (network, 5xx, timeout)
    """
    started_at = time.monotonic()

    check_cancelled(db, tenant_id, paper_id)
    update_status(db, tenant_id, paper_id, "ocr")

    # ── Download PDF from GCS ──
    try:
        pdf_bytes = download_bytes(storage_path)
    except FileNotFoundError as exc:
        raise FatalError(f"PDF not found at {storage_path}") from exc
    except Exception as exc:
        raise RetryableError(f"GCS download failed: {exc}") from exc

    if not pdf_bytes:
        raise FatalError(f"PDF empty at {storage_path}")

    logger.info(
        "ocr_download_done tenant=%s paper=%s bytes=%d", tenant_id, paper_id, len(pdf_bytes)
    )

    check_cancelled(db, tenant_id, paper_id)

    # ── R181 OCR cache check (SHA256 content hash) ──
    pdf_hash = _content_hash(pdf_bytes)
    cached = _try_load_cache(pdf_hash)
    if cached is not None:
        with contextlib.suppress(Exception):
            db.document(f"tenants/{tenant_id}/papers/{paper_id}").update(
                {"pageCount": cached.page_count, "ocrCacheHit": True, "ocrContentHash": pdf_hash}
            )
        elapsed_ms = int((time.monotonic() - started_at) * 1000)
        logger.info(
            "ocr_step_done tenant=%s paper=%s elapsed_ms=%d pages=%d cost_usd=0.0000 cache=hit",
            tenant_id, paper_id, elapsed_ms, cached.page_count,
        )
        return cached

    # ── OCR engine dispatch (R221) ──
    pages, engine_used = _run_ocr_engine(pdf_bytes)

    check_cancelled(db, tenant_id, paper_id)

    # ── Assemble result ──
    page_count = len(pages)
    full_text = "\n\n".join(p.text for p in pages)
    cost_usd = _engine_cost_usd(engine_used, page_count)
    latency_ms = int((time.monotonic() - started_at) * 1000)

    increment_cost(db, tenant_id, paper_id, "ocr", cost_usd)
    db.document(f"tenants/{tenant_id}/papers/{paper_id}").update({"pageCount": page_count})

    logger.info(
        "ocr_done tenant=%s paper=%s engine=%s pages=%d cost_usd=%.6f latency_ms=%d",
        tenant_id, paper_id, engine_used, page_count, cost_usd, latency_ms,
    )

    result = OcrResult(
        fullText=full_text,
        pages=pages,
        pageCount=page_count,
        costUsd=cost_usd,
    )
    _save_cache(pdf_hash, result)
    with contextlib.suppress(Exception):
        db.document(f"tenants/{tenant_id}/papers/{paper_id}").update(
            {"ocrCacheHit": False, "ocrContentHash": pdf_hash}
        )
    return result
