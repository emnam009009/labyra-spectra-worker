"""GCS wrapper: download raw bytes; lifecycle policy handles deletion."""

from __future__ import annotations

import logging
from io import BytesIO

from google.cloud import storage

from src.config import get_settings

logger = logging.getLogger(__name__)


def _client() -> storage.Client:
    return storage.Client(project=get_settings().gcp_project_id)


def parse_gs_url(gs_url: str) -> tuple[str, str]:
    """gs://bucket/path/to/file.xy → ('bucket', 'path/to/file.xy')."""
    if not gs_url.startswith("gs://"):
        raise ValueError(f"Not a gs:// URL: {gs_url}")
    rest = gs_url[5:]
    bucket, _, path = rest.partition("/")
    if not bucket or not path:
        raise ValueError(f"Malformed gs URL: {gs_url}")
    return bucket, path


def _resolve_bucket_and_path(location: str) -> tuple[str, str]:
    """Accept either full gs:// URL or relative path (with default bucket fallback).

    @phase R167-B8: relative path resolves bucket from FIREBASE_BUCKET env.
    Backward compat: gs:// URLs work as before.
    """
    if location.startswith("gs://"):
        return parse_gs_url(location)
    default_bucket = get_settings().firebase_bucket
    if not default_bucket:
        raise ValueError(
            f"Relative path '{location}' given but FIREBASE_BUCKET env not set"
        )
    return default_bucket, location.lstrip("/")


def download_bytes(location: str) -> bytes:
    """Download bytes from GCS. Accepts gs:// URL or relative path.

    @phase R167-B8
    """
    bucket_name, blob_path = _resolve_bucket_and_path(location)
    bucket = _client().bucket(bucket_name)
    blob = bucket.blob(blob_path)
    buf = BytesIO()
    blob.download_to_file(buf)
    data = buf.getvalue()
    logger.info("Downloaded %d bytes from gs://%s/%s", len(data), bucket_name, blob_path)
    return data


def download_text(gs_url: str, encoding: str = "utf-8") -> str:
    return download_bytes(gs_url).decode(encoding, errors="replace")


# ─── R181: OCR cache helpers @r181-applied ─────────────────────────
def upload_bytes(location: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    """Upload bytes to GCS. Accepts gs:// URL or relative path.

    Used by R181 OCR cache to persist Mistral OCR JSON results so that
    re-uploading the same PDF content (same SHA256) skips the OCR call.

    @phase R181
    """
    bucket_name, blob_path = _resolve_bucket_and_path(location)
    bucket = _client().bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_string(data, content_type=content_type)
    logger.info("Uploaded %d bytes to gs://%s/%s", len(data), bucket_name, blob_path)


def blob_exists(location: str) -> bool:
    """Check whether a blob exists in GCS. Accepts gs:// URL or relative path.

    @phase R181
    """
    bucket_name, blob_path = _resolve_bucket_and_path(location)
    bucket = _client().bucket(bucket_name)
    return bucket.blob(blob_path).exists()
