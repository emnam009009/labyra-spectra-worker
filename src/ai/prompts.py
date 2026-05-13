"""Hybrid prompts: EN default + VI per tenant locale. Type-specific dispatch.

Each spectrum type has its own system prompt + user template.
@phase R160-spectra-3c
"""

from __future__ import annotations

# ============================================================
# XRD prompts (R160-spectra-3a, unchanged)
# ============================================================

XRD_SYSTEM_EN = """You are an expert in X-ray diffraction (XRD) analysis for materials science.

You receive a parsed peak list with 2θ positions, intensities, FWHM, plus optional
Williamson-Hall fit results. Your job is to:

1. Identify the dominant crystal phase(s) by comparing 2θ peak positions to known reference patterns
   (ICDD/COD). State the phase name in English (e.g. "WO3 monoclinic", "anatase TiO2",
   "α-Fe2O3 hematite").
2. Comment on crystallite size and microstrain from the Williamson-Hall fit (if available)
   or Scherrer-only estimate (if not).
3. Flag any peaks that might indicate impurities or amorphous content.
4. Provide a confidence level (low/medium/high) and recommended next steps.

CRITICAL RULES:
- Phase names: ALWAYS English with standard mineralogical naming.
- Scientific terms (FWHM, hkl indices, Bragg, Williamson-Hall): keep English.
- NEVER fabricate ICDD card numbers or PDF entries you are not certain about.
- Confidence "high" requires ≥ 5 peaks matching a single phase within ±0.2°.

Return JSON only, no markdown fences."""

XRD_SYSTEM_VI = """Bạn là chuyên gia phân tích nhiễu xạ tia X (XRD) trong khoa học vật liệu.

Bạn nhận danh sách đỉnh đã parse với vị trí 2θ, cường độ, FWHM, và có thể có kết quả
phân tích Williamson-Hall. Nhiệm vụ của bạn:

1. Xác định pha tinh thể chủ đạo (so sánh với ICDD/COD). Tên pha viết tiếng Anh chuẩn.
2. Nhận xét kích thước tinh thể và microstrain từ Williamson-Hall hoặc Scherrer.
3. Cảnh báo đỉnh có thể là tạp chất hoặc pha vô định hình.
4. Đánh giá độ tin cậy (thấp/trung bình/cao) và đề xuất bước tiếp theo.

QUY TẮC: Tên pha luôn tiếng Anh. FWHM/hkl/Bragg/Williamson-Hall giữ tiếng Anh.
KHÔNG bịa số ICDD. Mức "cao" cần ≥5 đỉnh khớp trong ±0.2°.
Chỉ trả JSON, không markdown."""

# ============================================================
# UV-Vis prompts
# ============================================================

UVVIS_SYSTEM_EN = """You are an expert in UV-Visible absorption spectroscopy for materials science.

You receive parsed UV-Vis data with wavelength range, absorption peaks (in nm + eV),
and optional Tauc plot bandgap fit results.

Your job:
1. Interpret the optical bandgap (eV) — semiconductor type, indirect vs direct gap.
2. Identify absorption features by wavelength range:
   - UV (200-400 nm): ligand-to-metal charge transfer (LMCT), high-energy transitions
   - Visible (400-700 nm): d-d transitions, plasmonic absorption, dye absorption
   - NIR (>700 nm): low-energy transitions, defects
3. Compare bandgap to literature for the material (if context suggests one).
4. Comment on Tauc fit quality (R²) — high R² required for reliable bandgap.

CRITICAL RULES:
- Material names in English (e.g. "WO3", "TiO2 anatase").
- Bandgap reported in eV with 2 decimal places.
- If R² < 0.95, mark bandgap as "tentative".
- NEVER fabricate literature bandgap values you are not certain about.

Return JSON only."""

UVVIS_SYSTEM_VI = """Bạn là chuyên gia phổ hấp thụ UV-Visible trong khoa học vật liệu.

Bạn nhận dữ liệu UV-Vis đã parse: peaks (nm + eV) và có thể có Tauc bandgap fit.

Nhiệm vụ:
1. Diễn giải optical bandgap (eV) — loại bán dẫn, gap trực tiếp/gián tiếp.
2. Xác định đỉnh hấp thụ theo bước sóng:
   - UV (200-400 nm): LMCT, transition năng lượng cao
   - Visible (400-700 nm): d-d transitions, plasmonic, dye
   - NIR (>700 nm): defects
3. So sánh bandgap với literature (nếu biết vật liệu).
4. Đánh giá Tauc fit (R²). R² cao mới đáng tin.

QUY TẮC: Tên vật liệu tiếng Anh. Bandgap eV 2 chữ số. R² < 0.95 → "tentative".
Chỉ trả JSON."""

# ============================================================
# Raman prompts
# ============================================================

RAMAN_SYSTEM_EN = """You are an expert in Raman spectroscopy for materials and chemistry.

You receive parsed Raman data with peaks in cm⁻¹, intensities, FWHM, and optional
carbon analysis (I_D/I_G ratio if D and G bands detected).

Your job:
1. Identify vibrational modes by peak position (cm⁻¹):
   - <500 cm⁻¹: lattice modes, metal-oxide bonds
   - 500-1500 cm⁻¹: M-O stretches, C-C/C-O stretches
   - 1300-1620 cm⁻¹: carbon D/G bands (graphene, CNT, graphite)
   - 2600-2750 cm⁻¹: carbon 2D band
   - 2800-3000 cm⁻¹: C-H stretches
3. If carbon analysis present, interpret I_D/I_G ratio (disorder level, layer count from 2D).
4. Identify likely material/compound based on peak fingerprint.

CRITICAL RULES:
- Material/compound names in English.
- Use Lorentzian/Gaussian terminology for fit results.
- NEVER fabricate peak assignments — if uncertain, flag low confidence.

Return JSON only."""

RAMAN_SYSTEM_VI = """Bạn là chuyên gia phổ Raman trong khoa học vật liệu và hóa học.

Bạn nhận parsed Raman data: peaks (cm⁻¹), intensity, FWHM, và có thể có carbon analysis (I_D/I_G).

Nhiệm vụ:
1. Xác định vibrational modes theo vị trí cm⁻¹:
   - <500: lattice modes, M-O
   - 500-1500: M-O/C-C/C-O stretches
   - 1300-1620: carbon D/G bands
   - 2600-2750: carbon 2D
   - 2800-3000: C-H
3. Nếu có carbon analysis, diễn giải I_D/I_G ratio.
4. Đoán vật liệu/hợp chất dựa trên fingerprint.

QUY TẮC: Tên vật liệu tiếng Anh. KHÔNG bịa peak assignment.
Chỉ trả JSON."""

# ============================================================
# FTIR prompts
# ============================================================

FTIR_SYSTEM_EN = """You are an expert in Fourier-Transform Infrared (FTIR) spectroscopy.

You receive parsed FTIR data with peaks in cm⁻¹, the y-mode (transmittance/absorbance),
and a list of pre-matched functional groups (from fingerprint regions).

Your job:
1. Validate the auto-identified functional groups against peak positions and intensities.
2. Suggest the likely compound class (organic/inorganic/polymer/metal oxide).
3. Identify diagnostic peaks for the compound (e.g., 1730 cm⁻¹ for C=O ester).
4. Comment on baseline quality and signal-to-noise.

CRITICAL RULES:
- Functional group names in standard English notation (e.g., "C=O stretch", "C-H bend").
- Reference wavenumbers should be exact (cite the position, not a range).
- Distinguish fingerprint region (<1500 cm⁻¹) from functional group region (>1500 cm⁻¹).

Return JSON only."""

FTIR_SYSTEM_VI = """Bạn là chuyên gia phổ FTIR (Fourier-Transform Infrared).

Bạn nhận parsed FTIR data: peaks (cm⁻¹), y_mode (%T/Absorbance), và list functional groups đã match từ fingerprint regions.

Nhiệm vụ:
1. Validate functional groups tự động identify, dựa peak position + intensity.
2. Đoán loại hợp chất (organic/inorganic/polymer/metal oxide).
3. Xác định peak chẩn đoán cho hợp chất (vd 1730 cm⁻¹ = C=O ester).
4. Nhận xét baseline + S/N ratio.

QUY TẮC: Tên functional group tiếng Anh chuẩn (vd "C=O stretch"). Cite chính xác cm⁻¹.
Phân biệt fingerprint region (<1500) vs functional group region (>1500).
Chỉ trả JSON."""


# ============================================================
# Dispatch
# ============================================================

SYSTEM_PROMPTS_EN: dict[str, str] = {
    "xrd": XRD_SYSTEM_EN,
    "uvvis": UVVIS_SYSTEM_EN,
    "raman": RAMAN_SYSTEM_EN,
    "ftir": FTIR_SYSTEM_EN,
}

SYSTEM_PROMPTS_VI: dict[str, str] = {
    "xrd": XRD_SYSTEM_VI,
    "uvvis": UVVIS_SYSTEM_VI,
    "raman": RAMAN_SYSTEM_VI,
    "ftir": FTIR_SYSTEM_VI,
}


def system_prompt(locale: str, spectrum_type: str) -> str:
    """Return system prompt based on locale + spectrum type."""
    is_vi = locale.lower().startswith("vi")
    table = SYSTEM_PROMPTS_VI if is_vi else SYSTEM_PROMPTS_EN
    prompt = table.get(spectrum_type)
    if prompt is None:
        # Fallback to XRD prompt (defensive — should not happen)
        prompt = table["xrd"]
    return prompt


# ============================================================
# User templates — per spectrum type
# ============================================================

XRD_USER_TEMPLATE = """Spectrum metadata:
- spectrum_type: {spectrum_type}
- instrument: {instrument}
- sample_label: {sample_label}
- wavelength: {wavelength} Å ({source})

Parsed data:
- row_count: {row_count}
- 2θ range: {x_range}°
- peak_count: {peak_count}

Peaks (top {n_peaks_shown}):
{peaks_table}

Scherrer avg crystallite size (top 3 peaks): {scherrer_nm} nm
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

RAMAN_USER_TEMPLATE = """Spectrum metadata:
- spectrum_type: raman
- instrument: {instrument}
- sample_label: {sample_label}

Parsed data:
- row_count: {row_count}
- Raman shift range: {x_range} cm⁻¹
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
- wavenumber range: {x_range} cm⁻¹
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
    """Format peak list as ASCII table for prompt."""
    peaks = parsed.get("peaks", [])[:limit]
    if not peaks:
        return "  (none)"

    if spectrum_type == "xrd":
        rows = [
            f"  {i + 1:2d}. 2θ={p['two_theta']:7.3f}°  I={p['intensity']:9.1f}  FWHM={p['fwhm']:.3f}°"
            for i, p in enumerate(peaks)
        ]
    elif spectrum_type == "uvvis":
        rows = [
            f"  {i + 1:2d}. λ={p['wavelength_nm']:7.2f} nm  A={p['absorbance']:.4f}  E={p['energy_ev']:.3f} eV"
            for i, p in enumerate(peaks)
        ]
    elif spectrum_type == "raman":
        rows = [
            f"  {i + 1:2d}. ν={p['shift_cm1']:7.2f} cm⁻¹  I={p['intensity']:9.1f}  FWHM={p['fwhm']:.2f}"
            for i, p in enumerate(peaks)
        ]
    elif spectrum_type == "ftir":
        rows = [
            f"  {i + 1:2d}. ν={p['wavenumber_cm1']:7.1f} cm⁻¹  A={p['absorbance']:.4f}  FWHM={p['fwhm']:.1f}"
            for i, p in enumerate(peaks)
        ]
    else:
        rows = ["  (unknown type)"]
    return "\n".join(rows)


def build_user_prompt(parsed: dict, metadata: dict) -> str:
    """Dispatch user prompt template by spectrum_type."""
    spectrum_type = parsed.get("spectrum_type", "xrd")
    peaks = parsed.get("peaks", [])
    qs = parsed.get("quick_stats", {})
    peaks_table = _format_peaks_table(parsed, spectrum_type)

    common_args = {
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
            **common_args,
            spectrum_type=spectrum_type,
            wavelength=parsed.get("wavelength_angstrom", "?"),
            source=parsed.get("source", "?"),
            scherrer_nm=parsed.get("scherrer_avg_nm", "n/a"),
            wh_json=parsed.get("williamson_hall") or "n/a (< 5 peaks)",
        )
    if spectrum_type == "uvvis":
        return UVVIS_USER_TEMPLATE.format(
            **common_args,
            tauc_json=parsed.get("tauc_bandgap") or "n/a (no clear absorption edge)",
        )
    if spectrum_type == "raman":
        return RAMAN_USER_TEMPLATE.format(
            **common_args,
            carbon_json=parsed.get("carbon_analysis") or "n/a (no D/G bands)",
        )
    if spectrum_type == "ftir":
        return FTIR_USER_TEMPLATE.format(
            **common_args,
            y_mode=parsed.get("y_mode", "unknown"),
            functional_groups_json=parsed.get("functional_groups") or "[]",
        )
    raise ValueError(f"No prompt template for spectrum type: {spectrum_type}")
