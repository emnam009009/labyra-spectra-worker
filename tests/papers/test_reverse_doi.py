"""Tests for reverse DOI lookup by title via Crossref (R237cg)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

import src.papers.crossref as CR

_TITLE = "A review of the role and mechanism of surfactants in the morphology control of metal oxides"


def _mk(status, items=None):
    m = MagicMock()
    m.status_code = status
    m.json = MagicMock(return_value={"message": {"items": items or []}})
    return m


def _client(resp=None, side=None):
    c = patch.object(CR.httpx, "Client")
    cm = c.start()
    g = cm.return_value.__enter__.return_value.get
    if side is not None:
        g.side_effect = side
    else:
        g.return_value = resp
    return c


class TestReverseLookupDoi:
    def test_strong_match_returns_doi(self):
        c = _client(_mk(200, [{"DOI": "10.1039/d0nr12345", "title": [_TITLE]}]))
        try:
            assert CR.reverse_lookup_doi(_TITLE, ["Tran"], 2021) == "10.1039/d0nr12345"
        finally:
            c.stop()

    def test_weak_match_returns_none(self):
        c = _client(_mk(200, [{"DOI": "10.1/x", "title": ["Synthesis of perovskite solar cells"]}]))
        try:
            assert CR.reverse_lookup_doi(_TITLE) is None
        finally:
            c.stop()

    def test_picks_best_of_three(self):
        c = _client(_mk(200, [
            {"DOI": "10.1/a", "title": ["Unrelated A"]},
            {"DOI": "10.1039/correct", "title": [_TITLE]},
            {"DOI": "10.1/c", "title": ["Unrelated C"]},
        ]))
        try:
            assert CR.reverse_lookup_doi(_TITLE) == "10.1039/correct"
        finally:
            c.stop()

    def test_no_items(self):
        c = _client(_mk(200, []))
        try:
            assert CR.reverse_lookup_doi(_TITLE) is None
        finally:
            c.stop()

    def test_short_title_no_network(self):
        assert CR.reverse_lookup_doi("WO3") is None

    def test_network_error(self):
        c = _client(side=httpx.ConnectError("boom"))
        try:
            assert CR.reverse_lookup_doi(_TITLE) is None
        finally:
            c.stop()

    def test_non_200(self):
        c = _client(_mk(429, []))
        try:
            assert CR.reverse_lookup_doi(_TITLE) is None
        finally:
            c.stop()

    def test_query_shape(self):
        cap = {}
        c = patch.object(CR.httpx, "Client")
        cm = c.start()

        def grab(url, **kw):
            cap["u"] = url
            return _mk(200, [{"DOI": "10.1039/d0nr12345", "title": [_TITLE]}])

        cm.return_value.__enter__.return_value.get.side_effect = grab
        try:
            CR.reverse_lookup_doi(_TITLE, ["Tran"], 2021)
        finally:
            c.stop()
        assert "query.bibliographic=" in cap["u"]
        assert "select=DOI" in cap["u"]
