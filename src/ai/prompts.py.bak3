"""Hybrid prompts EN+VI per spectrum type. @phase R160-spectra-3c-hotfix"""

from __future__ import annotations

# ============================================================
# XRD prompts (unchanged)
# ============================================================

XRD_SYSTEM_EN = """You are an expert in X-ray diffraction (XRD) analysis for materials science.

You receive a parsed peak list with 2θ positions, intensities, FWHM, plus optional
Williamson-Hall fit results. Your job is to:

1. Identify the dominant crystal phase(s) by comparing 2θ peak positions to known reference patterns
   (ICDD/COD). State the phase name in English.
2. Comment on crystallite size and microstrain from the Williamson-Hall fit (if available)
   or Scherrer-only estimate (if not).
3. Flag any peaks that might indicate impurities or amorphous content.
4. Provide a confidence level (low/medium/high) and recommended next steps.

CRITICAL: Use plain ASCII for scientific units in JSON values:
- cm-1 (not cm⁻¹)
- sp2, sp3 (not sp² sp³)
- Frontend will format them prettily.

Return JSON only, no markdown fences."""

XRD_SYSTEM_VI = """Bạn là chuyên gia phân tích nhiễu xạ tia X (XRD).

Bạn nhận danh sách đỉnh với 2θ, cường độ, FWHM, và có thể có W-H fit. Nhiệm vụ:

1. Xác định pha tinh thể chủ đạo (ICDD/COD). Tên pha tiếng Anh.
2. Nhận xét kích thước tinh thể + microstrain.
3. Cảnh báo đỉnh tạp chất / amorphous.
4. Confidence + next steps.

CRITICAL: Dùng plain ASCII cho units trong JSON: cm-1, sp2, sp3 (KHÔNG cm⁻¹ sp²).
Chỉ trả JSON."""

# ============================================================
# UV-Vis prompts
# ============================================================

UVVIS_SYSTEM_EN = """You are an expert in UV-Visible absorption spectroscopy.

You receive parsed UV-Vis data with absorption peaks and Tauc plot bandgap fit.

Your job:
1. Interpret the optical bandgap (eV) — semiconductor type, direct vs indirect.
2. Identify absorption features by wavelength range (UV/Visible/NIR).
3. Compare to literature for the material (if known).
4. Comment on Tauc fit quality.

CRITICAL: Use plain ASCII units: cm-1, eV, nm. Frontend will format.
Return JSON only."""

UVVIS_SYSTEM_VI = """Chuyên gia phổ UV-Visible.

Bạn nhận peaks + Tauc bandgap fit. Nhiệm vụ:
1. Diễn giải bandgap (eV), direct/indirect.
2. Xác định absorption features theo bước sóng.
3. So sánh với literature.
4. Đánh giá Tauc fit.

CRITICAL: Plain ASCII units (cm-1, eV, nm). Chỉ trả JSON."""

# ============================================================
# UV-Vis DRS prompts (NEW)
# ============================================================

UVVIS_DRS_SYSTEM_EN = """You are an expert in UV-Vis Diffuse Reflectance Spectroscopy (DRS).

You receive parsed DRS data including reflectance curve, Kubelka-Munk F(R), and Tauc bandgap
computed on F(R) instead of absorbance. DRS is used for powders and opaque solids.

Your job:
1. Interpret the optical bandgap (eV) from Tauc-on-Kubelka-Munk.
2. Comment on the reflectance profile (high R indicates white/transparent, low R indicates absorbing).
3. Note that DRS bandgap may differ from transmission UV-Vis due to scattering.
4. Suggest sample type (typical powder photocatalyst, semiconductor pigment, etc.).

CRITICAL: Use plain ASCII units. Mention "Kubelka-Munk" explicitly when discussing F(R).
Return JSON only."""

UVVIS_DRS_SYSTEM_VI = """Chuyên gia UV-Vis Diffuse Reflectance Spectroscopy (DRS).

Bạn nhận reflectance curve + Kubelka-Munk F(R) + Tauc trên F(R). DRS dùng cho powder/opaque solids.

Nhiệm vụ:
1. Diễn giải bandgap (eV) từ Tauc-on-KM.
2. Nhận xét reflectance profile.
3. Lưu ý DRS bandgap có thể khác transmission UV-Vis do scattering.
4. Đoán loại mẫu (photocatalyst powder, pigment, etc.).

CRITICAL: Plain ASCII units. Nhắc "Kubelka-Munk" khi nói F(R). Chỉ trả JSON."""

# ============================================================
# Raman + FTIR prompts (unchanged from 3c)
# ============================================================

RAMAN_SYSTEM_EN = """You are an expert in Raman spectroscopy.

You receive peaks in cm-1, intensities, FWHM, and optional carbon D/G analysis.

Your job:
1. Identify vibrational modes by peak position (cm-1).
2. Interpret I_D/I_G if carbon analysis present.
3. Identify likely material from fingerprint.

CRITICAL: Plain ASCII units (cm-1, sp2, sp3). Return JSON only."""

RAMAN_SYSTEM_VI = """Chuyên gia Raman spectroscopy.

Bạn nhận peaks (cm-1), intensity, FWHM, optional carbon D/G analysis.

Nhiệm vụ:
1. Vibrational modes theo cm-1.
2. Diễn giải I_D/I_G nếu có.
3. Đoán vật liệu từ fingerprint.

CRITICAL: Plain ASCII units (cm-1, sp2, sp3). Chỉ trả JSON."""

FTIR_SYSTEM_EN = """You are an expert in FTIR spectroscopy.

You receive peaks in cm-1, y_mode, and pre-matched functional groups.

Your job:
1. Validate auto-identified functional groups.
2. Suggest compound class.
3. Identify diagnostic peaks.
4. Comment on baseline / S/N.

CRITICAL: Plain ASCII units (cm-1). Cite exact wavenumbers. Return JSON only."""

FTIR_SYSTEM_VI = """Chuyên gia FTIR.

Bạn nhận peaks (cm-1), y_mode, functional groups đã match.

Nhiệm vụ:
1. Validate functional groups.
2. Đoán compound class.
3. Diagnostic peaks.
4. Baseline + S/N.

CRITICAL: Plain ASCII units (cm-1). Cite chính xác wavenumber. Chỉ trả JSON."""


# ============================================================
# Dispatch
# ============================================================

SYSTEM_PROMPTS_EN: dict[str, str] = {
    "xrd": XRD_SYSTEM_EN,
    "uvvis": UVVIS_SYSTEM_EN,
    "uvvis_drs": UVVIS_DRS_SYSTEM_EN,
    "raman": RAMAN_SYSTEM_EN,
    "ftir": FTIR_SYSTEM_EN,
}

SYSTEM_PROMPTS_VI: dict[str, str] = {
    "xrd": XRD_SYSTEM_VI,
    "uvvis": UVVIS_SYSTEM_VI,
    "uvvis_drs": UVVIS_DRS_SYSTEM_VI,
    "raman": RAMAN_SYSTEM_VI,
    "ftir": FTIR_SYSTEM_VI,
}


def system_prompt(locale: str, spectrum_type: str) -> str:
    is_vi = locale.lower().startswith("vi")
    table = SYSTEM_PROMPTS_VI if is_vi else SYSTEM_PROMPTS_EN
    return table.get(spectrum_type, table["xrd"])


# ============================================================
# User templates
# ============================================================

XRD_USER_TEMPLATE = """Spectrum metadata:
- spectrum_type: {spectrum_type}
- instrument: {instrument}
- sample_label: {sample_label}
- wavelength: {wavelength} A ({source})

Parsed data:
- row_count: {row_count}
- 2theta range: {x_range} deg
- peak_count: {peak_count}

Peaks (top {n_peaks_shown}):
{peaks_table}

Scherrer avg crystallite size: {scherrer_nm} nm
Williamson-Hall: {wh_json}

Provide analysis as JSON:
{{
  "summary": "<3-sentence narrative in target language>",
  "phases": [{{"name": "<English>", "confidence": "low|medium|high", "matched_peaks": <int>, "note": "<localized>"}}],
  "crystallite_size_nm": <number | null>,
  "microstrain": <number | null>,
  "warnings": ["<localized>"],
  "next_steps": ["<localized>"],
  "overall_confidence": "low|medium|high"
}}"""

UVVIS_USER_TEMPLATE = """Spectrum metadata:
- spectrum_type: uvvis
- instrument: {instrument}
- sample_label: {sample_label}

Parsed data:
- row_count: {row_count}
- wavelength range: {x_range} nm
- peak_count: {peak_count}

Absorption peaks (top {n_peaks_shown}):
{peaks_table}

Tauc bandgap: {tauc_json}

Provide analysis as JSON:
{{
  "summary": "<3-sentence narrative in target language>",
  "bandgap": {{"value_ev": <number | null>, "transition": "direct|indirect|null", "confidence": "low|medium|high"}},
  "absorption_features": [{{"wavelength_nm": <number>, "assignment": "<English>", "note": "<localized>"}}],
  "warnings": ["<localized>"],
  "next_steps": ["<localized>"],
  "overall_confidence": "low|medium|high"
}}"""

UVVIS_DRS_USER_TEMPLATE = """Spectrum metadata:
- spectrum_type: uvvis_drs
- instrument: {instrument}
- sample_label: {sample_label}

Parsed data:
- row_count: {row_count}
- wavelength range: {x_range} nm
- reflectance_mode: {reflectance_mode}

Tauc bandgap (on Kubelka-Munk F(R)): {tauc_json}

Provide analysis as JSON:
{{
  "summary": "<3-sentence narrative in target language>",
  "bandgap": {{"value_ev": <number | null>, "transition": "direct|indirect|null", "confidence": "low|medium|high"}},
  "reflectance_profile": "<localized description>",
  "likely_sample_type": "<English | null>",
  "warnings": ["<localized>"],
  "next_steps": ["<localized>"],
  "overall_confidence": "low|medium|high"
}}"""

RAMAN_USER_TEMPLATE = """Spectrum metadata:
- spectrum_type: raman
- instrument: {instrument}
- sample_label: {sample_label}

Parsed data:
- row_count: {row_count}
- Raman shift range: {x_range} cm-1
- peak_count: {peak_count}

Peaks (top {n_peaks_shown}):
{peaks_table}

Carbon analysis: {carbon_json}

Provide analysis as JSON:
{{
  "summary": "<3-sentence narrative in target language>",
  "vibrational_modes": [{{"shift_cm1": <number>, "assignment": "<English>", "note": "<localized>"}}],
  "likely_material": "<English material name | null>",
  "carbon_interpretation": "<localized | null>",
  "warnings": ["<localized>"],
  "next_steps": ["<localized>"],
  "overall_confidence": "low|medium|high"
}}"""

FTIR_USER_TEMPLATE = """Spectrum metadata:
- spectrum_type: ftir
- instrument: {instrument}
- sample_label: {sample_label}

Parsed data:
- row_count: {row_count}
- wavenumber range: {x_range} cm-1
- peak_count: {peak_count}
- y_mode: {y_mode}

Peaks (top {n_peaks_shown}):
{peaks_table}

Auto-identified functional groups:
{functional_groups_json}

Provide analysis as JSON:
{{
  "summary": "<3-sentence narrative in target language>",
  "functional_groups": [{{"name": "<English>", "wavenumber_cm1": <number>, "confidence": "low|medium|high", "note": "<localized>"}}],
  "likely_compound_class": "<English | null>",
  "warnings": ["<localized>"],
  "next_steps": ["<localized>"],
  "overall_confidence": "low|medium|high"
}}"""


def _format_peaks_table(parsed: dict, spectrum_type: str, limit: int = 15) -> str:
    peaks = parsed.get("peaks", [])[:limit]
    if not peaks:
        return "  (none)"
    if spectrum_type == "xrd":
        return "\n".join(
            f"  {i + 1:2d}. 2theta={p['two_theta']:7.3f} deg  I={p['intensity']:9.1f}  FWHM={p['fwhm']:.3f}"
            for i, p in enumerate(peaks)
        )
    if spectrum_type == "uvvis":
        return "\n".join(
            f"  {i + 1:2d}. lambda={p['wavelength_nm']:7.2f} nm  A={p['absorbance']:.4f}  E={p['energy_ev']:.3f} eV"
            for i, p in enumerate(peaks)
        )
    if spectrum_type == "raman":
        return "\n".join(
            f"  {i + 1:2d}. nu={p['shift_cm1']:7.2f} cm-1  I={p['intensity']:9.1f}  FWHM={p['fwhm']:.2f}"
            for i, p in enumerate(peaks)
        )
    if spectrum_type == "ftir":
        return "\n".join(
            f"  {i + 1:2d}. nu={p['wavenumber_cm1']:7.1f} cm-1  A={p['absorbance']:.4f}  FWHM={p['fwhm']:.1f}"
            for i, p in enumerate(peaks)
        )
    return "  (unknown)"


def build_user_prompt(parsed: dict, metadata: dict) -> str:
    spectrum_type = parsed.get("spectrum_type", "xrd")
    peaks = parsed.get("peaks", [])
    qs = parsed.get("quick_stats", {})
    peaks_table = _format_peaks_table(parsed, spectrum_type)

    common = {
        "instrument": metadata.get("instrument", "(not specified)"),
        "sample_label": metadata.get("sampleLabel", "(unknown)"),
        "row_count": qs.get("rowCount", "?"),
        "x_range": qs.get("xRange", "?"),
        "peak_count": qs.get("peakCount", "?"),
        "n_peaks_shown": len(peaks[:15]),
        "peaks_table": peaks_table,
    }

    if spectrum_type == "xrd":
        return XRD_USER_TEMPLATE.format(
            **common, spectrum_type=spectrum_type,
            wavelength=parsed.get("wavelength_angstrom", "?"),
            source=parsed.get("source", "?"),
            scherrer_nm=parsed.get("scherrer_avg_nm", "n/a"),
            wh_json=parsed.get("williamson_hall") or "n/a",
        )
    if spectrum_type == "uvvis":
        return UVVIS_USER_TEMPLATE.format(
            **common, tauc_json=parsed.get("tauc_bandgap") or "n/a",
        )
    if spectrum_type == "uvvis_drs":
        return UVVIS_DRS_USER_TEMPLATE.format(
            **common,
            reflectance_mode=parsed.get("reflectance_mode", "?"),
            tauc_json=parsed.get("tauc_bandgap") or "n/a",
        )
    if spectrum_type == "raman":
        return RAMAN_USER_TEMPLATE.format(
            **common, carbon_json=parsed.get("carbon_analysis") or "n/a",
        )
    if spectrum_type == "ftir":
        return FTIR_USER_TEMPLATE.format(
            **common, y_mode=parsed.get("y_mode", "unknown"),
            functional_groups_json=parsed.get("functional_groups") or "[]",
        )
    raise ValueError(f"No template for: {spectrum_type}")
