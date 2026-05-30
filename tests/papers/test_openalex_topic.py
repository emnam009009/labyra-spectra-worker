"""Tests for fetch_openalex_topic — authoritative classification (R237bz)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

import src.papers.openalex as OA

_PAYLOAD = {
    "primary_topic": {
        "id": "https://openalex.org/T10024",
        "display_name": "Photocatalytic Water Splitting",
        "score": 0.9912,
        "subfield": {"id": "x", "display_name": "Electronic, Optical and Magnetic Materials"},
        "field": {"id": "25", "display_name": "Materials Science"},
        "domain": {"id": "3", "display_name": "Physical Sciences"},
    }
}


def _mk(status, js=None):
    m = MagicMock()
    m.status_code = status
    m.json = MagicMock(return_value=js or {})
    return m


def _with(resp):
    c = patch.object(OA.httpx, "Client")
    cm = c.start()
    cm.return_value.__enter__.return_value.get.return_value = resp
    return c


class TestFetchOpenAlexTopic:
    def test_full_path_parsed(self):
        c = _with(_mk(200, _PAYLOAD))
        try:
            t = OA.fetch_openalex_topic("10.1/x")
        finally:
            c.stop()
        assert t is not None
        assert t.topic == "Photocatalytic Water Splitting"
        assert t.field == "Materials Science"
        assert t.subfield == "Electronic, Optical and Magnetic Materials"
        assert t.domain == "Physical Sciences"
        assert abs(t.score - 0.9912) < 1e-6
        assert t.topic_id == "https://openalex.org/T10024"

    def test_404_returns_none(self):
        c = _with(_mk(404))
        try:
            assert OA.fetch_openalex_topic("10.1/x") is None
        finally:
            c.stop()

    def test_missing_primary_topic_returns_none(self):
        c = _with(_mk(200, {"id": "W1"}))
        try:
            assert OA.fetch_openalex_topic("10.1/x") is None
        finally:
            c.stop()

    def test_empty_doi_returns_none(self):
        assert OA.fetch_openalex_topic("") is None
        assert OA.fetch_openalex_topic("  ") is None

    def test_network_error_returns_none(self):
        c = patch.object(OA.httpx, "Client")
        cm = c.start()
        cm.return_value.__enter__.return_value.get.side_effect = httpx.ConnectError("boom")
        try:
            assert OA.fetch_openalex_topic("10.1/x") is None
        finally:
            c.stop()

    def test_url_has_select_and_key(self):
        captured = {}
        c = patch.object(OA.httpx, "Client")
        cm = c.start()

        def grab(url, **kw):
            captured["url"] = url
            return _mk(200, _PAYLOAD)

        cm.return_value.__enter__.return_value.get.side_effect = grab
        with patch.object(OA, "get_settings") as gs:
            gs.return_value.openalex_api_key = "oa-test"
            gs.return_value.openalex_polite_mailto = "x@y.z"
            gs.return_value.crossref_polite_mailto = ""
            try:
                OA.fetch_openalex_topic("10.1039/abc")
            finally:
                c.stop()
        assert "select=primary_topic" in captured["url"]
        assert "api_key=oa-test" in captured["url"]
        assert "doi:10.1039" in captured["url"]
