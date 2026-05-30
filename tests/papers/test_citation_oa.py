"""Tests for citation publisher + Open-Access enrichment (R237co)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

import src.papers.openalex as OA


def _mk(status, results):
    m = MagicMock()
    m.status_code = status
    m.json = MagicMock(return_value={"results": results})
    return m


def _client(resp=None, side=None):
    c = patch.object(OA.httpx, "Client")
    cm = c.start()
    g = cm.return_value.__enter__.return_value.get
    if side is not None:
        g.side_effect = side
    else:
        g.return_value = resp
    return c


_RESULTS = [
    {
        "doi": "https://doi.org/10.1039/D0NR07339C",
        "open_access": {"is_oa": True},
        "primary_location": {"source": {"host_organization_name": "Royal Society of Chemistry"}},
    },
    {
        "doi": "https://doi.org/10.1038/S43586-022-00164-0",
        "open_access": {"is_oa": False},
        "primary_location": {"source": {"host_organization_name": "Springer Nature"}},
    },
]


class TestOaBatch:
    def test_parses_publisher_and_oa(self):
        c = _client(_mk(200, _RESULTS))
        try:
            out = OA.fetch_openalex_oa_batch(
                ["10.1039/d0nr07339c", "10.1038/s43586-022-00164-0"]
            )
        finally:
            c.stop()
        assert out["10.1039/d0nr07339c"].publisher == "Royal Society of Chemistry"
        assert out["10.1039/d0nr07339c"].is_oa is True
        assert out["10.1038/s43586-022-00164-0"].is_oa is False

    def test_empty_input(self):
        assert OA.fetch_openalex_oa_batch([]) == {}

    def test_error_best_effort(self):
        c = _client(side=httpx.ConnectError("x"))
        try:
            assert OA.fetch_openalex_oa_batch(["10.1/a"]) == {}
        finally:
            c.stop()

    def test_non_200(self):
        c = _client(_mk(429, []))
        try:
            assert OA.fetch_openalex_oa_batch(["10.1/a"]) == {}
        finally:
            c.stop()

    def test_query_shape(self):
        cap = {}
        c = patch.object(OA.httpx, "Client")
        cm = c.start()

        def grab(url, **k):
            cap["u"] = url
            return _mk(200, [])

        cm.return_value.__enter__.return_value.get.side_effect = grab
        try:
            OA.fetch_openalex_oa_batch(["10.1/a", "10.1/b"])
        finally:
            c.stop()
        assert "filter=doi:" in cap["u"]
        assert "open_access" in cap["u"]


class TestCrossrefPublisher:
    def test_crossref_sets_publisher(self):
        import src.papers.crossref as CR

        msg = {
            "title": ["Some paper"],
            "publisher": "Elsevier BV",
            "container-title": ["Journal X"],
        }
        c = patch.object(CR.httpx, "Client")
        cm = c.start()
        r = MagicMock()
        r.status_code = 200
        r.json = MagicMock(return_value={"message": msg})
        cm.return_value.__enter__.return_value.get.return_value = r
        try:
            meta = CR.lookup_doi_crossref("10.1/x")
        finally:
            c.stop()
        assert meta is not None
        assert meta.publisher == "Elsevier BV"
        assert meta.is_open_access is None
