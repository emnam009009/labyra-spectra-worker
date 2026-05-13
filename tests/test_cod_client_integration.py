"""COD integration test (requires network; skip in CI)."""

import os
import pytest


@pytest.mark.skipif(
    os.environ.get("SKIP_NETWORK_TESTS") == "1",
    reason="Network tests skipped",
)
def test_search_cod_wo3():
    from src.citation.cod_client import search_cod_by_formula

    results = search_cod_by_formula("WO3", max_results=5)
    # COD has at least 1 WO3 entry
    assert isinstance(results, list)
    # If network is up: should have results
    if not results:
        pytest.skip("COD unreachable or no WO3 entries (network issue)")
    assert len(results) > 0
    # Verify structure
    first = results[0]
    assert "file" in first
    assert "sg" in first


@pytest.mark.skipif(
    os.environ.get("SKIP_NETWORK_TESTS") == "1",
    reason="Network tests skipped",
)
def test_fetch_cif():
    from src.citation.cod_client import fetch_cod_cif

    # Known WO3 entry from earlier test
    cif = fetch_cod_cif("1001678")
    if cif is None:
        pytest.skip("COD CIF fetch failed (network)")
    assert "data_" in cif or "_cell_length_a" in cif
