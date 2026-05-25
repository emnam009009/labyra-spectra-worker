"""
EU decimal-separator regression tests (worker bug audit B4).

EU-locale instruments (PerkinElmer/Bruker/Horiba) export numbers with a comma
decimal separator. The parsers used pandas to_numeric(default decimal='.'),
coercing "1,523" -> NaN -> dropna -> empty -> "no data parsed".

normalize_decimal() fixes this, but conservatively: it only rewrites when a
non-comma delimiter (tab/semicolon) is present, so ordinary comma-delimited
CSV (incl. integer columns like "400,1523") is never corrupted.
"""

from __future__ import annotations

from src.parsers._utils import normalize_decimal

# --- normalize_decimal unit behaviour ---------------------------------------

def test_us_comma_dot_untouched() -> None:
    """Standard US CSV (comma delimiter, dot decimal) is left alone."""
    txt = "23.5,1000\n24.0,1200"
    assert normalize_decimal(txt) == txt


def test_eu_tab_comma_converted() -> None:
    """EU tab-delimited comma-decimal -> dot."""
    out = normalize_decimal("400,5\t0,123\n402,0\t0,456")
    assert out == "400.5\t0.123\n402.0\t0.456"


def test_eu_semicolon_comma_converted() -> None:
    """EU semicolon-delimited comma-decimal -> dot."""
    out = normalize_decimal("400,5;0,123\n402,0;0,456")
    assert out == "400.5;0.123\n402.0;0.456"


def test_comma_delimited_integers_untouched() -> None:
    """CRITICAL: comma-delimited integer CSV must NOT be corrupted."""
    txt = "400,1523\n402,1601"
    assert normalize_decimal(txt) == txt


def test_comma_only_ambiguous_untouched() -> None:
    """Comma-only (no tab/;) is ambiguous -> left as-is (safe)."""
    txt = "400,5\n402,0"
    assert normalize_decimal(txt) == txt


def test_us_tab_dot_untouched() -> None:
    txt = "400.5\t0.123\n402.0\t0.456"
    assert normalize_decimal(txt) == txt


# --- end-to-end: parsers accept EU exports ----------------------------------

def test_ftir_eu_decimal_parses() -> None:
    """EU FTIR (tab + comma decimal, 400-4000 cm-1) parses, not empty."""
    from src.parsers.ftir import _parse_two_column

    # tab-delimited EU: wavenumber(comma-dec)  TAB  intensity(comma-dec)
    eu = "\n".join(f"{400 + i*4},{i%9}\t{0.1 + i*0.001:.3f}".replace(".", ",")
                   for i in range(200))
    x, _ = _parse_two_column(eu)
    assert len(x) > 10
    assert 300 < x.min() < 5000


def test_xrd_eu_decimal_parses() -> None:
    """EU XRD (tab + comma decimal) parses."""
    from src.parsers.xrd import _parse_two_column

    eu = "\n".join(f"{10 + i*0.1:.2f}\t{100 + i*0.5:.1f}".replace(".", ",")
                   for i in range(300))
    x, _ = _parse_two_column(eu)
    assert len(x) > 10
    assert 9 < x.min() < 11
