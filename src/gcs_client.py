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


def download_bytes(gs_url: str) -> bytes:
    bucket_name, blob_path = parse_gs_url(gs_url)
    bucket = _client().bucket(bucket_name)
    blob = bucket.blob(blob_path)
    buf = BytesIO()
    blob.download_to_file(buf)
    data = buf.getvalue()
    logger.info("Downloaded %d bytes from %s", len(data), gs_url)
    return data


def download_text(gs_url: str, encoding: str = "utf-8") -> str:
    return download_bytes(gs_url).decode(encoding, errors="replace")
