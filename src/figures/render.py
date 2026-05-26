"""Publication-grade spectrum figure rendering with matplotlib.

Renders a spectrum (line + optional peak markers) at a target publisher's exact
figure dimensions, resolution, typography, and line weight, then returns the
encoded bytes in the requested vector or raster format. This is the export path
that complements the interactive Plotly chart in the web app: Plotly is for
on-screen exploration; this produces the file that goes into a manuscript.

Design choices follow common high-impact-journal figure conventions: a single
clean axis frame, ticks pointing out, no chartjunk, sans-serif labels at the
journal's required point size, and a prominent (>=1 pt) data line. The figure is
sized in physical units (mm -> inches) so the on-page width is exactly the
column width; matplotlib then renders at the preset DPI.

@phase R261 (publication figure export).
"""

from __future__ import annotations

import io
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: no display, safe on Cloud Run
import matplotlib.pyplot as plt

from src.figures.presets import MM_PER_INCH, FigurePreset, get_preset

# Axis labels per spectrum type (x reversed for FTIR by convention).
_AXES: dict[str, dict[str, Any]] = {
    "xrd": {"x": "2θ (degrees)", "y": "Intensity (counts)", "reverse_x": False},
    "raman": {"x": "Raman shift (cm⁻¹)", "y": "Intensity (a.u.)", "reverse_x": False},
    "ftir": {"x": "Wavenumber (cm⁻¹)", "y": "Transmittance (%)", "reverse_x": True},
    "uvvis": {"x": "Wavelength (nm)", "y": "Absorbance", "reverse_x": False},
    "lsv": {"x": "Potential (V vs RHE)", "y": "j (mA cm⁻²)", "reverse_x": False},
    "cv": {"x": "Potential (V)", "y": "Current (A)", "reverse_x": False},
}


def _peak_xy(spectrum_type: str, peaks: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    """Extract (x, y) of peaks by spectrum type."""
    xkey = {
        "xrd": "two_theta",
        "raman": "shift_cm1",
        "ftir": "wavenumber_cm1",
        "uvvis": "wavelength_nm",
    }.get(spectrum_type)
    ykey = {
        "xrd": "intensity",
        "raman": "intensity",
        "ftir": "absorbance",
        "uvvis": "absorbance",
    }.get(spectrum_type)
    if not xkey or not ykey:
        return [], []
    xs = [p[xkey] for p in peaks if xkey in p and ykey in p]
    ys = [p[ykey] for p in peaks if xkey in p and ykey in p]
    return xs, ys


def render_spectrum_figure(
    *,
    spectrum_type: str,
    curve: dict[str, list[float]],
    peaks: list[dict[str, Any]] | None = None,
    publisher: str = "nature",
    column: str = "single",
    fmt: str = "pdf",
    line_color: str = "#1f4e9c",
    show_peaks: bool = True,
    peak_labels: list[str] | None = None,
    title: str | None = None,
    height_ratio: float = 0.62,
) -> tuple[bytes, str]:
    """Render a spectrum to publication-grade bytes.

    Returns (data, mime_type). ``curve`` must have ``x`` and ``y`` lists.
    ``fmt`` is one of png/pdf/svg/eps/tiff. The figure width equals the
    publisher's column width; height is width * height_ratio (clamped to the
    preset max height). Vector formats (pdf/svg/eps) embed editable text; raster
    formats (png/tiff) use the preset DPI.
    """
    preset: FigurePreset = get_preset(publisher)
    x = curve.get("x") or []
    y = curve.get("y") or []
    if not x or not y:
        raise ValueError("curve must contain non-empty x and y arrays")

    axes_cfg = _AXES.get(spectrum_type, {"x": "X", "y": "Y", "reverse_x": False})

    width_in = preset.width_in(column)
    height_in = width_in * height_ratio
    if preset.max_height_mm:
        height_in = min(height_in, preset.max_height_mm / MM_PER_INCH)

    # Resolution: vector formats ignore DPI for geometry; line-heavy spectra use
    # the line-art DPI for raster so thin features stay crisp.
    dpi = preset.line_dpi if fmt in ("png", "tiff") else preset.min_dpi

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": preset.font_family,
            "font.size": preset.font_size_pt,
            "axes.linewidth": 0.8,
            "axes.labelsize": preset.font_size_pt,
            "xtick.labelsize": preset.font_size_pt - 1,
            "ytick.labelsize": preset.font_size_pt - 1,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "svg.fonttype": "none",  # keep text editable in SVG
            "pdf.fonttype": 42,  # TrueType (editable) in PDF/PS
            "ps.fonttype": 42,
        }
    ):
        fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
        ax.plot(x, y, color=line_color, linewidth=preset.line_width_pt)

        if show_peaks and peaks:
            px, py = _peak_xy(spectrum_type, peaks)
            if px:
                ax.scatter(px, py, marker="v", s=14, color="#b22222", zorder=5,
                           linewidths=0.4, edgecolors="white")
                if peak_labels:
                    for i, (xx, yy) in enumerate(zip(px, py, strict=False)):
                        if i < len(peak_labels) and peak_labels[i]:
                            ax.annotate(
                                peak_labels[i], (xx, yy),
                                textcoords="offset points", xytext=(0, 4),
                                ha="center", fontsize=preset.font_size_pt - 2,
                            )

        ax.set_xlabel(axes_cfg["x"])
        ax.set_ylabel(axes_cfg["y"])
        if title:
            ax.set_title(title, fontsize=preset.font_size_pt)
        if axes_cfg.get("reverse_x"):
            ax.invert_xaxis()

        # Minimal frame: drop top/right spines (clean journal look).
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(width=0.8)
        # Keep the figure at the exact column width (do NOT use bbox_inches=tight,
        # which shrinks the canvas below the journal column spec). Reserve margins
        # for labels via subplots_adjust instead.
        fig.subplots_adjust(left=0.16, right=0.97, top=0.93 if title else 0.97, bottom=0.16)

        buf = io.BytesIO()
        save_kwargs: dict[str, Any] = {"format": "tiff" if fmt == "tiff" else fmt}
        if fmt in ("png", "tiff"):
            save_kwargs["dpi"] = dpi
        if fmt == "tiff":
            save_kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(buf, **save_kwargs)
        plt.close(fig)

    mime = {
        "png": "image/png",
        "pdf": "application/pdf",
        "svg": "image/svg+xml",
        "eps": "application/postscript",
        "tiff": "image/tiff",
    }.get(fmt, "application/octet-stream")
    return buf.getvalue(), mime
