"""Unit tests for profile cache.

@phase R185-4f-profile-cache
"""
from __future__ import annotations

import time

from src.deviation.profile_cache import TTLCache


class TestBasicCaching:
    def test_first_call_fetches(self):
        cache = TTLCache(ttl_seconds=60)
        calls = []

        def fetcher(key):
            calls.append(key)
            return {"formula": key, "data": "test"}

        result = cache.get_or_fetch("MoS2", fetcher)
        assert result["formula"] == "MoS2"
        assert calls == ["MoS2"]

    def test_second_call_uses_cache(self):
        cache = TTLCache(ttl_seconds=60)
        calls = []

        def fetcher(key):
            calls.append(key)
            return {"formula": key}

        cache.get_or_fetch("MoS2", fetcher)
        cache.get_or_fetch("MoS2", fetcher)
        cache.get_or_fetch("MoS2", fetcher)
        assert calls == ["MoS2"]  # only fetched once

    def test_different_keys_fetch_separately(self):
        cache = TTLCache(ttl_seconds=60)
        calls = []

        def fetcher(key):
            calls.append(key)
            return {"formula": key}

        cache.get_or_fetch("MoS2", fetcher)
        cache.get_or_fetch("WS2", fetcher)
        cache.get_or_fetch("MoS2", fetcher)
        assert calls == ["MoS2", "WS2"]


class TestTTL:
    def test_expired_entry_refetched(self):
        cache = TTLCache(ttl_seconds=1)
        calls = []

        def fetcher(key):
            calls.append(key)
            return {"formula": key}

        cache.get_or_fetch("MoS2", fetcher)
        time.sleep(1.1)
        cache.get_or_fetch("MoS2", fetcher)
        assert len(calls) == 2


class TestNoneNotCached:
    def test_none_returns_not_cached(self):
        cache = TTLCache(ttl_seconds=60)
        calls = []

        def fetcher(key):
            calls.append(key)
            return None

        cache.get_or_fetch("Unknown", fetcher)
        cache.get_or_fetch("Unknown", fetcher)
        # None values should re-fetch every time
        assert len(calls) == 2


class TestEviction:
    def test_max_size_eviction(self):
        cache = TTLCache(ttl_seconds=60, max_size=5)

        def fetcher(key):
            return {"formula": key}

        for i in range(10):
            cache.get_or_fetch(f"M{i}", fetcher)

        stats = cache.stats()
        assert stats["size"] <= 5


class TestStats:
    def test_hits_misses_tracked(self):
        cache = TTLCache(ttl_seconds=60)

        def fetcher(key):
            return {"formula": key}

        cache.get_or_fetch("MoS2", fetcher)
        cache.get_or_fetch("MoS2", fetcher)
        cache.get_or_fetch("WS2", fetcher)

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["hit_rate"] == round(1 / 3, 3)


class TestInvalidate:
    def test_invalidate_drops_key(self):
        cache = TTLCache(ttl_seconds=60)
        calls = []

        def fetcher(key):
            calls.append(key)
            return {"formula": key}

        cache.get_or_fetch("MoS2", fetcher)
        cache.invalidate("MoS2")
        cache.get_or_fetch("MoS2", fetcher)
        assert len(calls) == 2

    def test_clear_drops_all(self):
        cache = TTLCache(ttl_seconds=60)

        def fetcher(key):
            return {"formula": key}

        cache.get_or_fetch("A", fetcher)
        cache.get_or_fetch("B", fetcher)
        cache.clear()
        assert cache.stats()["size"] == 0
        assert cache.stats()["hits"] == 0
