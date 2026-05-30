"""Paper processing pipeline orchestrator.

Port labyra-app/src/lib/ai/rag/pipeline/orchestrator.ts.

Single entry point: process_paper(). Runs full pipeline OCR → chunk →
[enrich] → embed → index → citation, with state machine updates and
cost accounting per step.

@phase R167-B6
"""
from __future__ import annotations

import logging
import time
from typing import Any

from google.cloud import firestore  # type: ignore[import-untyped]
from google.cloud.firestore_v1 import SERVER_TIMESTAMP  # type: ignore[import-untyped]

from src.papers.chunking import chunk_paper
from src.papers.citation import run_citation_step
from src.papers.embed import run_embed_step
from src.papers.enrich import run_enrich_step
from src.papers.errors import CancelledError, FatalError, RetryableError
from src.papers.index import run_index_step
from src.papers.google_books import lookup_book_isbn, search_book_by_title
from src.papers.metadata import extract_metadata
from src.papers.ocr import run_ocr_step
from src.papers.state import (
    check_cancelled,
    load_paper,
    set_cancelled,
    set_error,
    update_status,
)
from src.papers.types import PaperDoc

logger = logging.getLogger(__name__)

# Fatal error patterns — match TS isFatalError
_FATAL_ERROR_PATTERNS = (
    "unauthorized",
    "quota_exceeded",
    "invalid_pdf",
    "malformed",
)


def _is_fatal_error_message(msg: str) -> bool:
    """Check if an error message indicates a permanent (non-retryable) failure."""
    lower = msg.lower()
    return any(p in lower for p in _FATAL_ERROR_PATTERNS)


def _log_event(event: str, **fields: Any) -> None:
    """Structured JSON-friendly log via standard logger (Cloud Logging auto-indexes)."""
    logger.info("%s %s", event, " ".join(f"{k}={v}" for k, v in fields.items()))


def process_paper(
    db: firestore.Client,
    tenant_id: str,
    paper_id: str,
    storage_path: str,
    job_id: str,
    created_by: str,
) -> None:
    """Run full paper processing pipeline.

    Idempotency contract (Pub/Sub at-least-once delivery):
      - Paper already 'indexed' → return early (no work)
      - Paper 'cancelling'/'cancelled' → set_cancelled() + return
      - Each step itself idempotent (Firestore .set overwrites, Pinecone
        upsert overwrites by ID, createCitation dedup by deterministic ID)

    Args:
        db: Firestore client
        tenant_id, paper_id: identity
        storage_path: GCS path to PDF (from Pub/Sub message)
        job_id: unique job ID for trace correlation
        created_by: UID of user who triggered (for ProvBase on citations)

    Raises:
        CancelledError: user cancelled — handler should 204 ack
        FatalError: permanent failure — handler should 400 ack
        RetryableError: transient failure — handler should 500 (Pub/Sub retry)
    """
    started_at = time.monotonic()
    _log_event("pipeline_start", tenant=tenant_id, paper=paper_id, job=job_id)

    # ── Load paper + idempotency check ──────────────────────
    try:
        paper = load_paper(db, tenant_id, paper_id)
    except FatalError:
        raise  # paper not found = no point retrying
    except Exception as exc:
        raise RetryableError(f"load_paper failed: {exc}") from exc

    # Already indexed (Pub/Sub duplicate delivery after success)
    if paper.status == "indexed":
        _log_event("pipeline_skip_already_indexed", paper=paper_id)
        return

    # User-initiated cancellation already recorded
    if paper.status in ("cancelling", "cancelled"):
        _log_event("pipeline_already_cancelled", paper=paper_id)
        set_cancelled(db, tenant_id, paper_id)
        return

    # Mark processing start (replaces any previous attempt timestamp)
    db.document(f"tenants/{tenant_id}/papers/{paper_id}").update({
        "processingStartedAt": SERVER_TIMESTAMP,
    })

    try:
        # ── STEP 1: OCR ─────────────────────────────────────
        _log_event("step_ocr_start", paper=paper_id)
        ocr_result = run_ocr_step(db, tenant_id, paper_id, storage_path)
        _log_event(
            "step_ocr_done",
            paper=paper_id, pages=ocr_result.page_count, cost_usd=ocr_result.cost_usd,
        )

        # ── STEP 1b: Metadata extract (best-effort, non-blocking) ──
        # Mirrors TS hotfix-5d-4 — extract real title/year/DOI from page 1
        # R177-1d: also detects documentType + isbn + publisher (for books)
        first_page_text = ocr_result.pages[0].text if ocr_result.pages else ""
        try:
            # @r179-8-applied: log input + output for debug
            logger.info(
                "step1b_input tenant=%s paper=%s page0_len=%d page0_preview=%r",
                tenant_id, paper_id, len(first_page_text), first_page_text[:200],
            )
            meta = extract_metadata(first_page_text)
            logger.info(
                "step1b_output tenant=%s paper=%s title=%r authors=%d year=%s doi=%r",
                tenant_id, paper_id, meta.title, len(meta.authors), meta.year, meta.doi,
            )
            # R237bm (gap B): if Gemini found no DOI, scan pages 1-3 for a
            # LABELLED self-DOI before giving up. Deterministic, best-effort —
            # the DOI is printed on the opening pages of most articles.
            self_doi_source = "gemini" if (meta.doi or "").strip() else ""
            if not (meta.doi or "").strip():
                from src.papers.self_doi_resolver import extract_self_doi

                pages_text = [p.text for p in ocr_result.pages[:3]]
                recovered = extract_self_doi(pages_text)
                if recovered.found:
                    meta.doi = recovered.doi
                    self_doi_source = recovered.source
                    logger.info(
                        "self_doi_recovered tenant=%s paper=%s doi=%r source=%s",
                        tenant_id, paper_id, recovered.doi, recovered.source,
                    )
            # Persist input length + output snapshot to Firestore for offline debug
            try:
                db.document(f"tenants/{tenant_id}/papers/{paper_id}").update({
                    "_debugStep1b": {
                        "page0Len": len(first_page_text),
                        "page0Preview": first_page_text[:300],
                        "extractedTitle": meta.title,
                        "extractedAuthors": meta.authors[:3] if meta.authors else [],
                        "extractedYear": meta.year,
                        "extractedDoi": meta.doi,
                    },
                })
            except Exception:
                pass
            update_payload = {
                "title": meta.title,
                "authors": meta.authors,
                "year": meta.year,
                "doi": meta.doi,
                "documentType": meta.document_type,
                "isbn": meta.isbn,
                "publisher": meta.publisher,
                "selfDoiSource": self_doi_source,
                "metadataExtractedAt": SERVER_TIMESTAMP,
            }
            db.document(f"tenants/{tenant_id}/papers/{paper_id}").update(update_payload)
            # Update local paper for downstream index step (uses real metadata)
            paper = PaperDoc.model_validate({
                **paper.model_dump(by_alias=True),
                "title": meta.title,
                "authors": meta.authors,
                "year": meta.year,
                "doi": meta.doi,
                "documentType": meta.document_type,
                "isbn": meta.isbn,
                "publisher": meta.publisher,
            })
            _log_event(
                "metadata_extracted",
                paper=paper_id,
                title_chars=len(meta.title),
                authors=len(meta.authors),
                document_type=meta.document_type,
            )
        except Exception as exc:  # noqa: BLE001 — non-fatal
            # @r179-6-applied: persist exception detail to Firestore for debug
            error_repr = f"{type(exc).__name__}: {exc}"[:300]
            _log_event("metadata_extract_skipped", paper=paper_id, error=error_repr[:100])
            try:
                db.document(f"tenants/{tenant_id}/papers/{paper_id}").update({
                    "metadataExtractError": error_repr,
                })
            except Exception:
                pass  # debug field is best-effort
            meta = None  # used by Step 1c

        # ── STEP 1c: Book metadata resolve (R177-1d, best-effort) ──
        # If documentType=book, query Google Books for canonical title/year/
        # publisher/page_count. Crossref+OpenAlex don\'t index books.
        # Skipped silently when documentType != "book" or Step 1b failed.
        if meta and meta.document_type == "book":
            try:
                book = None
                # Path 1: ISBN exact (highest confidence)
                if meta.isbn:
                    book = lookup_book_isbn(meta.isbn)
                # Path 2: title fuzzy fallback (Jaccard ≥ 0.8 inside resolver)
                if book is None and meta.title and meta.title != "Untitled":
                    book = search_book_by_title(meta.title, meta.authors or None)

                if book is not None:
                    # Merge canonical fields from Google Books, prefer
                    # API values over OCR-extracted (more reliable).
                    book_update = {
                        "title": book.title or meta.title,
                        "year": book.year or meta.year,
                        "publisher": book.publisher or meta.publisher,
                        "isbn": book.isbn_13 or book.isbn_10 or meta.isbn,
                        "bookPageCount": book.page_count,
                        "bookSubtitle": book.subtitle,
                        "bookSourceId": book.source_id,
                        "bookResolvedAt": SERVER_TIMESTAMP,
                    }
                    db.document(f"tenants/{tenant_id}/papers/{paper_id}").update(book_update)
                    _log_event(
                        "book_metadata_resolved",
                        paper=paper_id,
                        source_id=book.source_id,
                        publisher=book.publisher[:40],
                        year=book.year,
                    )
                else:
                    _log_event(
                        "book_metadata_unresolved",
                        paper=paper_id,
                        had_isbn=bool(meta.isbn),
                        title=meta.title[:60],
                    )
            except Exception as exc:  # noqa: BLE001 — non-fatal
                _log_event(
                    "book_metadata_skipped",
                    paper=paper_id,
                    error=str(exc)[:100],
                )

        # ── STEP 1d: Domain classification (R178-3) ────────────
        # @r178-3-applied
        # @r179-5-applied: re-indented INSIDE try block (was orphan 4-space)
        # Best-effort: failure → fallback unknown, never blocks pipeline.
        # Audit log: tenants/{tid}/_audit_classify/{paperId}_{ts}
        try:
            from src.papers.classify import classify_paper_domain
            import time as _time

            classify_result = classify_paper_domain(ocr_result.full_text)
            now_ms = int(_time.time() * 1000)

            db.document(f"tenants/{tenant_id}/papers/{paper_id}").update({
                "domain": classify_result.classification.primary,
                "subtopics": classify_result.classification.subtopics,
                "domainConfidence": classify_result.classification.confidence.value,
                "domainClassifiedAt": now_ms,
                "domainModelVersion": classify_result.model_version,
                "domainPromptVersion": classify_result.prompt_version,
                "domainTaxonomyVersion": classify_result.taxonomy_version,
            })

            audit_id = f"{paper_id}_{now_ms}"
            audit_doc = {
                "paperId": paper_id,
                "classifiedAt": SERVER_TIMESTAMP,
                "modelVersion": classify_result.model_version,
                "promptVersion": classify_result.prompt_version,
                "taxonomyVersion": classify_result.taxonomy_version,
                "inputTokens": classify_result.input_tokens,
                "outputTokens": classify_result.output_tokens,
                "costUsd": classify_result.cost_usd,
                "result": {
                    "primary": classify_result.classification.primary,
                    "subtopics": classify_result.classification.subtopics,
                    "confidence": classify_result.classification.confidence.value,
                    "reasoning": classify_result.classification.reasoning,
                },
            }
            if classify_result.rejected:
                audit_doc["rejected"] = {
                    "reason": classify_result.rejected_reason,
                    "rawResponse": classify_result.raw_response[:2000],
                }
            db.document(
                f"tenants/{tenant_id}/_audit_classify/{audit_id}"
            ).set(audit_doc)

            logger.info(
                "step1d_classify_done tenant=%s paper=%s primary=%s rejected=%s cost=%.6f",
                tenant_id, paper_id,
                classify_result.classification.primary,
                classify_result.rejected, classify_result.cost_usd,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                "step1d_classify_failed tenant=%s paper=%s err=%s — continuing pipeline",
                tenant_id, paper_id, exc,
            )

        # ── STEP 1e: Journal metadata resolution (R179-2) ────
        # @r179-2-applied
        # @r179-5-applied: re-indented INSIDE try block
        try:
            from src.papers.journal_resolve import resolve_journal_from_doi

            doi_value = ""
            if meta is not None and hasattr(meta, "doi"):
                doi_value = meta.doi or ""

            if doi_value:
                journal_result = resolve_journal_from_doi(doi_value)
                journal_update = {
                    "journal": journal_result.journal,
                    "journalShort": journal_result.journal_short,
                    "journalIssn": journal_result.journal_issn,
                    "journalSourceId": journal_result.source_id,
                    "journalResolvedAt": journal_result.resolved_at,
                }
                # R228 + R237bm (gap A/C): the publisher's title is authoritative
                # for OCR typos (e.g. 'Phage'→'Please'), BUT a hallucinated/wrong
                # DOI would resolve to a DIFFERENT paper. So only override when the
                # multi-tier guard accepts the record; otherwise keep the OCR title
                # and flag it for manual review. When accepted, also adopt the
                # canonical authors (gap C).
                if journal_result.title:
                    from src.papers.self_doi_resolver import should_override_title

                    do_override, reason = should_override_title(
                        meta.title if meta is not None else "",
                        meta.authors if meta is not None else [],
                        journal_result.title,
                        journal_result.authors,
                    )
                    if do_override:
                        journal_update["title"] = journal_result.title
                        journal_update["titleSourceId"] = journal_result.source_id
                        journal_update["doiTitleMismatch"] = False
                        local_update: dict = {"title": journal_result.title}
                        if journal_result.authors:
                            journal_update["authors"] = journal_result.authors
                            local_update["authors"] = journal_result.authors
                        try:
                            paper = paper.model_copy(update=local_update)
                        except Exception:  # noqa: BLE001 — best-effort local sync
                            pass
                        logger.info(
                            "step1e_title_override tenant=%s paper=%s reason=%s ocr=%r -> %s=%r",
                            tenant_id, paper_id, reason,
                            (meta.title if meta is not None else "")[:60],
                            journal_result.source_id,
                            journal_result.title[:60],
                        )
                    else:
                        # DOI resolved to a title that doesn't match — keep OCR
                        # title, flag for manual confirmation (Trust > Coverage).
                        journal_update["doiTitleMismatch"] = True
                        logger.warning(
                            "step1e_title_mismatch tenant=%s paper=%s reason=%s "
                            "ocr=%r resolved=%s=%r (kept OCR title)",
                            tenant_id, paper_id, reason,
                            (meta.title if meta is not None else "")[:60],
                            journal_result.source_id,
                            journal_result.title[:60],
                        )
                db.document(f"tenants/{tenant_id}/papers/{paper_id}").update(journal_update)
                logger.info(
                    "step1e_journal_done tenant=%s paper=%s journal=%s source=%s rejected=%s",
                    tenant_id, paper_id,
                    journal_result.journal or "(none)",
                    journal_result.source_id or "(none)",
                    journal_result.rejected,
                )
            else:
                logger.info(
                    "step1e_journal_skip tenant=%s paper=%s reason=no_doi",
                    tenant_id, paper_id,
                )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                "step1e_journal_failed tenant=%s paper=%s err=%s — continuing pipeline",
                tenant_id, paper_id, exc,
            )

        # ── STEP 2: Chunking ────────────────────────────────
        update_status(db, tenant_id, paper_id, "chunking")
        _log_event("step_chunking_start", paper=paper_id)
        chunks = chunk_paper(ocr_result)
        db.document(f"tenants/{tenant_id}/papers/{paper_id}").update({
            "chunkCount": len(chunks),
        })
        _log_event("step_chunking_done", paper=paper_id, chunks=len(chunks))

        if not chunks:
            raise FatalError("no chunks extracted — empty or malformed OCR output")

                # ── STEP 3: Contextual enrichment (default OFF) ─────
        update_status(db, tenant_id, paper_id, "enriching")
        _log_event("step_enriching_start", paper=paper_id)
        enriched = run_enrich_step(db, tenant_id, paper_id, ocr_result.full_text, chunks)
        db.document(f"tenants/{tenant_id}/papers/{paper_id}").update({
            "enrichedChunkCount": len(enriched),
        })
        _log_event("step_enriching_done", paper=paper_id, enriched=len(enriched))

        # ── STEP 4: Embedding ───────────────────────────────
        update_status(db, tenant_id, paper_id, "embedding")
        _log_event("step_embedding_start", paper=paper_id)
        embedded = run_embed_step(db, tenant_id, paper_id, enriched)
        db.document(f"tenants/{tenant_id}/papers/{paper_id}").update({
            "embeddedChunkCount": len(embedded),
        })
        _log_event("step_embedding_done", paper=paper_id, embedded=len(embedded))

        # ── STEP 5: Indexing ────────────────────────────────
        update_status(db, tenant_id, paper_id, "indexing")
        _log_event("step_indexing_start", paper=paper_id)
        indexed_count = run_index_step(db, tenant_id, paper, embedded)
        db.document(f"tenants/{tenant_id}/papers/{paper_id}").update({
            "indexedChunkCount": indexed_count,
        })
        _log_event("step_indexing_done", paper=paper_id, indexed=indexed_count)

        # ── STEP 6: Citation extraction (non-blocking) ──────
        # Per ADR-017: citation extraction is enhancement, not blocker. Paper
        # is still searchable via vector embeddings even if citation fails.
        try:
            update_status(db, tenant_id, paper_id, "extracting_citations")
            _log_event("step_citation_start", paper=paper_id)
            citation_result = run_citation_step(
                db, tenant_id, paper_id,
                created_by=created_by or paper.created_by,
                full_text=ocr_result.full_text,
                self_doi=(meta.doi or "").strip() or None,
            )
            _log_event(
                "step_citation_done",
                paper=paper_id,
                dois_found=citation_result.dois_found,
                citations_created=citation_result.citations_created,
                resolutions_linked=citation_result.resolutions_linked,
                api_failures=citation_result.api_failures,
            )
        except CancelledError:
            raise  # cancellation always propagates
        except Exception as exc:  # noqa: BLE001 — citation non-fatal
            _log_event(
                "step_citation_failed", paper=paper_id, error=str(exc)[:200],
            )

        # ── DONE ────────────────────────────────────────────
        total_latency_ms = int((time.monotonic() - started_at) * 1000)
        update_status(db, tenant_id, paper_id, "indexed", extra_fields={
            "processingCompletedAt": SERVER_TIMESTAMP,
            "totalLatencyMs": total_latency_ms,
        })
        _log_event("pipeline_complete", paper=paper_id, latency_ms=total_latency_ms)

    except CancelledError:
        _log_event("pipeline_cancelled", paper=paper_id)
        set_cancelled(db, tenant_id, paper_id)
        raise

    except FatalError as exc:
        msg = str(exc)
        _log_event("pipeline_fatal_error", paper=paper_id, error=msg[:200])
        set_error(db, tenant_id, paper_id, msg, is_retryable=False)
        raise

    except RetryableError as exc:
        msg = str(exc)
        # Check if the underlying error is actually fatal by message pattern
        if _is_fatal_error_message(msg):
            _log_event("pipeline_fatal_pattern", paper=paper_id, error=msg[:200])
            set_error(db, tenant_id, paper_id, msg, is_retryable=False)
            raise FatalError(msg) from exc
        _log_event("pipeline_retryable_error", paper=paper_id, error=msg[:200])
        set_error(db, tenant_id, paper_id, msg, is_retryable=True)
        raise

    except Exception as exc:  # noqa: BLE001 — final catch-all
        msg = str(exc) or exc.__class__.__name__
        if _is_fatal_error_message(msg):
            _log_event("pipeline_unexpected_fatal", paper=paper_id, error=msg[:200])
            set_error(db, tenant_id, paper_id, msg, is_retryable=False)
            raise FatalError(msg) from exc
        _log_event("pipeline_unexpected_error", paper=paper_id, error=msg[:200])
        set_error(db, tenant_id, paper_id, msg, is_retryable=True)
        # Re-raise as RetryableError so Pub/Sub retries (handler returns 500)
        raise RetryableError(msg) from exc
