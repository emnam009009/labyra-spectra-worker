"""Datalab hosted Marker OCR — async POST + poll (httpx).

Datalab's PAID cloud service (https://www.datalab.to/api/v1/marker), NOT self-hosted
weights → the Chandra OpenRAIL $2M gate does not apply here (paying per page).

Flow: POST /api/v1/marker (multipart, X-Api-Key) -> {request_id, request_check_url}
      -> poll GET request_check_url until status == "complete" -> {markdown, ...}

Config (src/config.py Settings):
    datalab_api_key        required
    datalab_marker_url     default https://www.datalab.to/api/v1/marker
    datalab_use_llm        higher accuracy, small hallucination risk, slower (default False)
    datalab_langs          optional OCR languages, e.g. "English,Vietnamese"

@phase R221
"""
from __future__ import annotations

import base64
import logging
import re
import time
from typing import NamedTuple

import httpx

from src.config import get_settings
from src.papers.errors import FatalError, RetryableError
from src.papers.types import OcrPage


class RawFigure(NamedTuple):
    """A figure decoded from the OCR response, ready to upload to storage."""

    name: str
    page: int
    mime_type: str
    data: bytes


_FIG_REF_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def _build_raw_figures(pages: list[OcrPage], images: dict[str, str]) -> list[RawFigure]:
    """Decode Marker's {name: base64} images and map each to the page that
    references it in the paginated markdown. Best-effort — bad entries skipped."""
    page_of: dict[str, int] = {}
    for pg in pages:
        for match in _FIG_REF_RE.finditer(pg.text):
            page_of.setdefault(match.group(1).strip(), pg.page_number)
    figures: list[RawFigure] = []
    for name, b64 in images.items():
        try:
            data = base64.b64decode(b64)
        except (ValueError, TypeError):
            continue
        mime = "image/png" if name.lower().endswith(".png") else "image/jpeg"
        figures.append(RawFigure(name=name, page=page_of.get(name, 0), mime_type=mime, data=data))
    return figures

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 2.0
_MAX_POLLS = 180  # ~6 min ceiling
_SUBMIT_TIMEOUT_S = 60.0
_POLL_TIMEOUT_S = 30.0

# Marker paginate delimiter (verified): "\n\n{N}" + 48 dashes + "\n\n", N 0-indexed.
_PAGE_MARKER = re.compile(r"\{(\d+)\}-{48}")


def _split_pages(markdown: str) -> list[OcrPage]:
    """Split Marker paginated markdown into per-page OcrPage list.

    No markers found (format change) -> whole blob as one page so OCR still works
    (coarse page attribution).
    """
    markers = list(_PAGE_MARKER.finditer(markdown))
    if not markers:
        return [OcrPage(pageNumber=1, text=markdown.strip())]
    pages: list[OcrPage] = []
    for i, match in enumerate(markers):
        start = match.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(markdown)
        pages.append(
            OcrPage(pageNumber=int(match.group(1)) + 1, text=markdown[start:end].strip())
        )
    return pages


def datalab_ocr(pdf_bytes: bytes) -> tuple[list[OcrPage], list[RawFigure]]:
    """Run Datalab Marker OCR on PDF bytes. Returns (pages, figures).

    Raises:
        FatalError: missing/invalid API key (401).
        RetryableError: network / 5xx / timeout / conversion error.
    """
    settings = get_settings()
    if not settings.datalab_api_key:
        raise FatalError("DATALAB_API_KEY missing in worker settings")

    headers = {"X-Api-Key": settings.datalab_api_key}
    data: dict[str, str] = {"output_format": "markdown", "paginate": "true"}
    if settings.datalab_use_llm:
        data["use_llm"] = "true"
    if settings.datalab_langs:
        data["langs"] = settings.datalab_langs
    files = {"file": ("document.pdf", pdf_bytes, "application/pdf")}

    # ── Submit (async; returns request_check_url) ──
    try:
        with httpx.Client(timeout=_SUBMIT_TIMEOUT_S) as client:
            resp = client.post(
                settings.datalab_marker_url, headers=headers, data=data, files=files
            )
    except httpx.HTTPError as exc:
        raise RetryableError(f"Datalab submit network error: {exc}") from exc

    if resp.status_code == 401:
        raise FatalError("Datalab 401 — invalid DATALAB_API_KEY")
    if resp.status_code >= 400:
        raise RetryableError(f"Datalab submit failed: {resp.status_code} {resp.text[:300]}")

    submit = resp.json()
    check_url = submit.get("request_check_url")
    if not submit.get("success") or not check_url:
        raise RetryableError(
            f"Datalab submit error: {submit.get('error') or 'no request_check_url'}"
        )

    # ── Poll until complete ──
    for _ in range(_MAX_POLLS):
        time.sleep(_POLL_INTERVAL_S)
        try:
            with httpx.Client(timeout=_POLL_TIMEOUT_S) as client:
                poll = client.get(check_url, headers=headers)
        except httpx.HTTPError as exc:
            raise RetryableError(f"Datalab poll network error: {exc}") from exc
        if poll.status_code >= 400:
            raise RetryableError(f"Datalab poll failed: {poll.status_code} {poll.text[:200]}")
        result = poll.json()
        if result.get("status") == "complete":
            if not result.get("success"):
                raise RetryableError(
                    f"Datalab conversion failed: {result.get('error') or 'unknown'}"
                )
            pages = _split_pages(result.get("markdown") or "")
            images = result.get("images")
            figures = _build_raw_figures(pages, images) if isinstance(images, dict) else []
            logger.info("datalab_ocr_done pages=%d figures=%d", len(pages), len(figures))
            return pages, figures

    raise RetryableError(f"Datalab timed out after {int(_MAX_POLLS * _POLL_INTERVAL_S)}s")
