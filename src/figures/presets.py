"""Publisher figure-size presets for publication-grade spectrum export.

Each preset encodes the official author-guideline specs for figure dimensions,
typography, resolution, and accepted formats. Specs verified from primary
publisher sources (see docs/scientific-methods/figure-export.md for links and
verification dates). Specs differ between individual journals within a publisher
family, so these are the *typical* values; the UI surfaces a "verify the target
journal" note.

@phase R261 (publication figure export).
"""

from __future__ import annotations

from dataclasses import dataclass

MM_PER_INCH = 25.4


@dataclass(frozen=True)
class FigurePreset:
    """One publisher's figure specification."""

    key: str
    label: str
    single_col_mm: float
    double_col_mm: float
    max_height_mm: float | None
    font_family: list[str]  # preference order; first available wins
    font_size_pt: float  # body/axis label size at final print size
    min_dpi: int  # raster minimum (halftone)
    line_dpi: int  # raster line-art minimum
    line_width_pt: float  # prominent plot-line weight
    preferred_formats: list[str]  # vector/raster the publisher accepts
    note: str = ""

    def width_mm(self, column: str) -> float:
        return self.double_col_mm if column == "double" else self.single_col_mm

    def width_in(self, column: str) -> float:
        return self.width_mm(column) / MM_PER_INCH


# Sans-serif stack: publishers uniformly want Arial/Helvetica. DejaVu Sans is
# the metrically-near fallback bundled with matplotlib so rendering never fails.


PRESETS: dict[str, FigurePreset] = {
    # Nature — research-figure-guide.nature.com (verified 2026-05)
    "nature": FigurePreset(
        key="nature",
        label="Nature",
        single_col_mm=89.0,
        double_col_mm=183.0,
        max_height_mm=170.0,
        font_family=["Arial", "Helvetica", "DejaVu Sans"],
        font_size_pt=7.0,  # Nature requires 5-7 pt; 7 is the legible upper bound
        min_dpi=300,
        line_dpi=600,
        line_width_pt=1.0,
        preferred_formats=["pdf", "eps", "svg", "tiff", "png"],
        note="Fonts must be 5-7 pt; max height 170 mm. Vector (PDF/EPS/SVG) preferred.",
    ),
    # ACS — pubs.acs.org / researcher-resources.acs.org (verified 2026-05)
    "acs": FigurePreset(
        key="acs",
        label="ACS (American Chemical Society)",
        single_col_mm=82.6,  # 3.25 in
        double_col_mm=178.0,  # 7 in
        max_height_mm=233.0,  # 9.17 in
        font_family=["Helvetica", "Arial", "DejaVu Sans"],
        font_size_pt=8.0,  # ACS: no smaller than 8 pt (some journals 5 pt min)
        min_dpi=300,
        line_dpi=600,
        line_width_pt=1.0,  # ACS: lines no thinner than 1 pt
        preferred_formats=["eps", "pdf", "tiff", "png"],
        note="Lines >=1 pt, font >=8 pt. Submit EPS/TIFF/PDF (not raw PPT).",
    ),
    # Elsevier — elsevier.com artwork instructions (verified 2026-05)
    "elsevier": FigurePreset(
        key="elsevier",
        label="Elsevier",
        single_col_mm=90.0,
        double_col_mm=190.0,
        max_height_mm=None,
        font_family=["Arial", "Helvetica", "DejaVu Sans"],
        font_size_pt=8.0,
        min_dpi=300,  # halftone; combination 500, line art 1000
        line_dpi=1000,
        line_width_pt=1.0,  # prominent plot lines ~1 pt; min 0.25 pt
        preferred_formats=["eps", "pdf", "tiff", "png"],
        note="Defaults 90/190 mm — verify the specific journal. Line art 1000 dpi.",
    ),
    # RSC — rsc.org figures guidelines (verified 2026-05)
    "rsc": FigurePreset(
        key="rsc",
        label="RSC (Royal Society of Chemistry)",
        single_col_mm=83.0,
        double_col_mm=171.0,
        max_height_mm=233.0,
        font_family=["Arial", "Helvetica", "DejaVu Sans"],
        font_size_pt=8.0,
        min_dpi=600,
        line_dpi=600,
        line_width_pt=1.0,
        preferred_formats=["tiff", "eps", "pdf", "png"],
        note="TIFF >=600 dpi (or EPS/PDF, converted to TIFF). Max height 233 mm.",
    ),
}

DEFAULT_PRESET = "nature"
VALID_COLUMNS = ("single", "double")
VALID_FORMATS = ("png", "pdf", "svg", "eps", "tiff")


def get_preset(key: str) -> FigurePreset:
    """Return the preset for key, falling back to the default."""
    return PRESETS.get(key.strip().lower(), PRESETS[DEFAULT_PRESET])
