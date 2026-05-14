"""Citation cache abstraction layer.

Migration-safe Protocol pattern:
- FirestoreCitationCache (current implementation)
- Future: RedisCitationCache, PostgresCitationCache

Cache stores expensive computations: CIF text + Dans_Diffraction simulated peaks.
Hit rate >80% expected (same materials repeated: Cu, WO3, TiO2, ZnO, Fe2O3, ...).

Path convention (any backend): {source}-{entry_id} key.
TTL: 30 days.

@phase R161-cache
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 30 * 24 * 3600  # 30 days


def _normalize_key(source: str, entry_id: str) -> str:
    """Sanitize cache key (safe for Firestore doc id, Redis key, SQL row)."""
    safe_id = str(entry_id).replace("/", "_").replace(".", "_")
    return f"{source.lower()}-{safe_id}"


class CitationCache(Protocol):
    """Cache interface for citation lookup results."""

    def get(self, source: str, entry_id: str) -> dict[str, Any] | None:
        """Return cached payload or None if missing/expired."""
        ...

    def set(self, source: str, entry_id: str, payload: dict[str, Any]) -> None:
        """Store payload under (source, entry_id) key."""
        ...


# ============================================================
# Firestore implementation
# ============================================================
class FirestoreCitationCache:
    """Firestore-backed citation cache.

    Path: tenants/_global/citation_cache/{source}-{entry_id}
    Schema: {cif_text, simulated_peaks, lattice_params, cached_at}
    """

    COLLECTION = "citation_cache"

    def __init__(self, project_id: str):
        from google.cloud import firestore
        self._project_id = project_id
        self._db: firestore.Client | None = None

    def _client(self):
        if self._db is None:
            from google.cloud import firestore
            self._db = firestore.Client(project=self._project_id)
        return self._db

    def _doc_ref(self, key: str):
        return (
            self._client()
            .collection("tenants")
            .document("_global")
            .collection(self.COLLECTION)
            .document(key)
        )

    def get(self, source: str, entry_id: str) -> dict[str, Any] | None:
        try:
            key = _normalize_key(source, entry_id)
            doc = self._doc_ref(key).get()
            if not doc.exists:
                return None
            data = doc.to_dict() or {}
            cached_at = data.get("cached_at", 0)
            if time.time() - cached_at > CACHE_TTL_SECONDS:
                return None
            return data
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cache read failed for %s/%s: %s", source, entry_id, exc)
            return None

    def set(self, source: str, entry_id: str, payload: dict[str, Any]) -> None:
        try:
            key = _normalize_key(source, entry_id)
            self._doc_ref(key).set({**payload, "cached_at": time.time()})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cache write failed for %s/%s: %s", source, entry_id, exc)


# ============================================================
# No-op (for tests / when cache disabled)
# ============================================================
class NoOpCitationCache:
    """Disable cache (always miss)."""

    def get(self, source: str, entry_id: str) -> dict[str, Any] | None:  # noqa: ARG002
        return None

    def set(self, source: str, entry_id: str, payload: dict[str, Any]) -> None:  # noqa: ARG002
        pass


# ============================================================
# Module-level singleton
# ============================================================
_cache: CitationCache | None = None


def get_cache() -> CitationCache:
    """Return module-level cache instance.

    Backed by Firestore in production; falls back to no-op if init fails.
    """
    global _cache
    if _cache is None:
        try:
            from src.config import get_settings
            _cache = FirestoreCitationCache(project_id=get_settings().gcp_project_id)
            logger.info("Citation cache initialized: FirestoreCitationCache")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Citation cache init failed, falling back to no-op: %s", exc)
            _cache = NoOpCitationCache()
    return _cache


def set_cache(cache: CitationCache) -> None:
    """Override cache (mainly for testing)."""
    global _cache
    _cache = cache
