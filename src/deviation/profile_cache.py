"""
TTL LRU cache for materialProfile reads.

Worker-process-local cache. Each Cloud Run instance has its own cache.
At scale (autoscaled instances), hit rate stays high because:
  - 20 hot materials cover ~95% of queries
  - Each instance handles ~10 concurrent → cache warms quickly
  - 60-min TTL absorbs profile updates

Thread-safe via threading.Lock (FastAPI sync handlers share process).

@phase R185-4f-profile-cache
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from time import time
from typing import Any

logger = logging.getLogger(__name__)


class TTLCache:
    """Simple TTL LRU cache with hit/miss stats."""

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 200):
        self._cache: dict[str, tuple[float, Any]] = {}
        self._ttl = ttl_seconds
        self._max = max_size
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get_or_fetch(self, key: str, fetch_fn: Callable[[str], Any]) -> Any:
        """Return cached value if fresh, else fetch + cache."""
        now = time()

        # Fast path: check cache under lock
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                ts, value = cached
                if now - ts < self._ttl:
                    self._hits += 1
                    return value
                # Stale → drop
                del self._cache[key]

        # Cache miss → fetch outside lock (don\'t block readers)
        try:
            value = fetch_fn(key)
        except Exception:
            logger.exception("Cache fetch_fn failed for key=%s", key)
            raise

        # Store under lock
        with self._lock:
            self._misses += 1
            if value is not None:
                self._cache[key] = (now, value)
                self._evict_if_needed()

        return value

    def _evict_if_needed(self) -> None:
        """Drop oldest 20% when at capacity. Caller holds lock."""
        if len(self._cache) <= self._max:
            return
        sorted_items = sorted(self._cache.items(), key=lambda x: x[1][0])
        drop_count = max(1, self._max // 5)
        for k, _ in sorted_items[:drop_count]:
            self._cache.pop(k, None)

    def invalidate(self, key: str) -> None:
        """Drop a specific entry (e.g., after admin update)."""
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
                "ttl_seconds": self._ttl,
                "max_size": self._max,
            }


# ── Singleton for materialProfile cache ──────────────────────────────────────
_profile_cache = TTLCache(ttl_seconds=3600, max_size=200)


def cached_get_material_profile(
    formula: str,
    fetch_fn: Callable[[str], dict[str, Any] | None],
) -> dict[str, Any] | None:
    """Cached wrapper around get_material_profile."""
    return _profile_cache.get_or_fetch(formula, fetch_fn)


def cache_stats() -> dict[str, Any]:
    """Return cache statistics."""
    return _profile_cache.stats()


def cache_invalidate(formula: str) -> None:
    """Drop a specific formula from cache."""
    _profile_cache.invalidate(formula)


def cache_clear() -> None:
    """Drop all cached entries."""
    _profile_cache.clear()
