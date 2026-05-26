# Publication Figure Export (Publisher Presets)

Module: `src/figures/` · Phase: R261 · Endpoint: `POST /render-figure`

## Purpose

Render a spectrum (XRD/Raman/FTIR/UV-Vis/LSV/CV) to a manuscript-ready figure
file whose physical dimensions, resolution, typography, line weight, and format
match a target publisher's author guidelines. This complements the interactive
Plotly chart in the web app: Plotly is for on-screen exploration; matplotlib
(server-side) produces the file that goes into a paper, which is the convention
for high-impact-journal figures.

## Why matplotlib server-side, not Plotly client-side

High-impact journals expect vector figures (PDF/EPS/SVG) with embedded editable
text, exact column widths, and controlled font sizes. matplotlib gives precise
physical sizing (mm → inch → DPI) and editable-text vector output; the worker
already runs Python with numpy/scipy, so the spectrum curve is rendered next to
where it is parsed.

## Publisher specifications (verified primary sources)

All widths are printed-figure column widths. Specs differ between individual
journals within a publisher family — these are the typical values, and the
export UI shows a "verify the specific journal" note.

| Publisher | Single col | Double col | Max height | Font | Min DPI (halftone) | Line-art DPI | Vector |
|-----------|-----------|-----------|-----------|------|-----|------|--------|
| Nature    | 89 mm     | 183 mm    | 170 mm    | Arial/Helvetica 5–7 pt | 300 | 600 | PDF/EPS/SVG |
| ACS       | 82.6 mm (3.25 in) | 178 mm (7 in) | 233 mm | Helvetica/Arial ≥8 pt | 300 | 600 | EPS/PDF |
| Elsevier  | 90 mm     | 190 mm    | —         | Arial/Helvetica | 300 | 1000 | EPS/PDF |
| RSC       | 83 mm     | 171 mm    | 233 mm    | Arial/Helvetica | 600 | 600 | TIFF/EPS/PDF |

Common to all: prominent plot lines ≥1 pt; sans-serif labels; vector preferred,
raster (TIFF/PNG) at the stated DPI as fallback. Greek/symbol characters kept as
text (not paths) for editability.

### Sources (verified 2026-05)

- Nature: https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/
  (89/183 mm; max height 170 mm; fonts 5–7 pt).
- ACS: https://pubs.acs.org/page/4authors/submission/graphics_prep.html and
  https://researcher-resources.acs.org/publish/author_guidelines
  (single 3.25 in; double 4.167–7 in; lines ≥1 pt; fonts ≥8 pt; EPS/TIFF/PDF).
- Elsevier: https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions
  (defaults ~90/190 mm; 300 halftone / 500 combination / 1000 line-art DPI;
  prominent lines ~1 pt, min 0.25 pt).
- RSC: https://www.rsc.org/journals-books-databases/author-and-reviewer-hub/authors-information/prepare-and-format/figures-graphics-images/
  (single 8.3 cm; double 17.1 cm; max 23.3 cm; TIFF ≥600 dpi or EPS/PDF).

## Implementation

`src/figures/presets.py` — `FigurePreset` dataclass + `PRESETS` dict + `get_preset`.
`src/figures/render.py` — `render_spectrum_figure(...)`:

- figure width = preset column width (mm → inch); height = width × 0.62, clamped
  to the preset max height;
- DPI = line-art DPI for raster (png/tiff), min DPI for vector geometry;
- `rc_context` sets the sans-serif stack (Arial → Helvetica → DejaVu Sans
  fallback so rendering never fails), font size, line width, and
  `svg.fonttype='none'` / `pdf.fonttype=42` to keep text editable;
- minimal frame (top/right spines hidden, ticks out), optional peak markers and
  per-peak labels (e.g. functional-group names for FTIR);
- exact column width preserved by `subplots_adjust` (not `bbox_inches='tight'`,
  which would shrink below the journal spec).

Output formats: `png`, `pdf`, `svg`, `eps`, `tiff` (LZW).

## Endpoint

`POST /render-figure` (stateless, no auth — like `/reference/parse`):

```json
{
  "spectrum_type": "ftir",
  "curve": {"x": [...], "y": [...]},
  "peaks": [{"wavenumber_cm1": 3439, "absorbance": 0.5}],
  "publisher": "nature",
  "column": "single",
  "fmt": "pdf",
  "peak_labels": ["O-H stretch"],
  "line_color": "#1f4e9c",
  "title": null
}
```

Returns the file inline with `Content-Disposition: attachment`.

## Caveats

- Column-width specs vary by individual journal within a family; always confirm
  against the target journal's current author guidelines before final submission.
- EPS from matplotlib is widely accepted but does not support transparency; use
  PDF/SVG when transparency is needed.
- True CMYK conversion (some print workflows) is out of scope; output is RGB.
