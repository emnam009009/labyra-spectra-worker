"""Tests for publication-grade figure export (src.figures)."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from src.figures.presets import MM_PER_INCH, PRESETS, get_preset
from src.figures.render import render_spectrum_figure


def _ftir_curve() -> dict[str, list[float]]:
    x = np.linspace(400, 4000, 800)
    y = 100 - 30 * np.exp(-((x - 3439) ** 2) / 2000) - 20 * np.exp(-((x - 1108) ** 2) / 1500)
    return {"x": x.tolist(), "y": y.tolist()}


def test_all_presets_have_required_fields() -> None:
    for key, p in PRESETS.items():
        assert p.key == key
        assert p.single_col_mm > 0 and p.double_col_mm > p.single_col_mm
        assert p.min_dpi >= 300
        assert p.preferred_formats


def test_get_preset_fallback() -> None:
    assert get_preset("nonsense").key == "nature"  # default
    assert get_preset("ACS").key == "acs"  # case-insensitive


@pytest.mark.parametrize("publisher,expected_mm", [
    ("nature", 89.0), ("acs", 82.6), ("elsevier", 90.0), ("rsc", 83.0),
])
def test_png_width_matches_column_spec(publisher: str, expected_mm: float) -> None:
    """Rendered PNG physical width must equal the journal column width."""
    data, mime = render_spectrum_figure(
        spectrum_type="ftir", curve=_ftir_curve(),
        publisher=publisher, column="single", fmt="png",
    )
    assert mime == "image/png"
    img = Image.open(io.BytesIO(data))
    w_px = img.size[0]
    dpi = img.info.get("dpi", (0,))[0]
    assert dpi > 0
    width_mm = w_px / dpi * MM_PER_INCH
    assert abs(width_mm - expected_mm) < 1.0  # within 1 mm


def test_double_column_wider_than_single() -> None:
    single, _ = render_spectrum_figure(
        spectrum_type="ftir", curve=_ftir_curve(),
        publisher="nature", column="single", fmt="png",
    )
    double, _ = render_spectrum_figure(
        spectrum_type="ftir", curve=_ftir_curve(),
        publisher="nature", column="double", fmt="png",
    )
    assert Image.open(io.BytesIO(double)).size[0] > Image.open(io.BytesIO(single)).size[0]


@pytest.mark.parametrize("fmt,mime", [
    ("pdf", "application/pdf"),
    ("svg", "image/svg+xml"),
    ("eps", "application/postscript"),
    ("png", "image/png"),
    ("tiff", "image/tiff"),
])
def test_formats_render(fmt: str, mime: str) -> None:
    data, got_mime = render_spectrum_figure(
        spectrum_type="ftir", curve=_ftir_curve(),
        publisher="nature", column="single", fmt=fmt,
    )
    assert got_mime == mime
    assert len(data) > 0


def test_svg_keeps_text_editable() -> None:
    data, _ = render_spectrum_figure(
        spectrum_type="ftir", curve=_ftir_curve(),
        publisher="nature", column="single", fmt="svg",
    )
    # svg.fonttype='none' keeps labels as <text>, not paths
    assert b"<text" in data


def test_empty_curve_raises() -> None:
    with pytest.raises(ValueError):
        render_spectrum_figure(
            spectrum_type="ftir", curve={"x": [], "y": []},
            publisher="nature", column="single", fmt="png",
        )


def test_peak_labels_render() -> None:
    data, _ = render_spectrum_figure(
        spectrum_type="ftir", curve=_ftir_curve(),
        peaks=[{"wavenumber_cm1": 3439, "absorbance": 0.5}],
        publisher="nature", column="single", fmt="svg",
        peak_labels=["O-H stretch"], show_peaks=True,
    )
    assert b"O-H" in data or b"stretch" in data
