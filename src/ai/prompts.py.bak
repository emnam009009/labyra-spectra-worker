"""Hybrid prompt strategy: English default + Vietnamese fallback per tenant.

Rule (R160-spectra-3a):
- locale == "vi" → use Vietnamese system prompt
- everything else → English system prompt
- Scientific terms (phase names, hkl, FWHM, Tauc) always English even in VI output
- JSON schema fields named in English; narrative fields localized

Extending:
- Add more locales here. Falls back to "en" if locale not found.
"""

from __future__ import annotations

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
- NEVER fabricate ICDD card numbers or PDF entries you are not certain about. If you cite
  a reference, mark it as "tentative" unless 2θ positions match within ±0.1°.
- Confidence "high" requires ≥ 5 peaks matching a single phase within ±0.2°.

Return JSON only, no markdown fences."""

XRD_SYSTEM_VI = """Bạn là chuyên gia phân tích nhiễu xạ tia X (XRD) trong khoa học vật liệu.

Bạn nhận danh sách đỉnh đã parse với vị trí 2θ, cường độ, FWHM, và có thể có kết quả
phân tích Williamson-Hall. Nhiệm vụ của bạn:

1. Xác định pha tinh thể chủ đạo bằng cách so sánh vị trí đỉnh 2θ với các mẫu chuẩn
   (ICDD/COD). Tên pha viết bằng tiếng Anh chuẩn (vd "WO3 monoclinic", "anatase TiO2",
   "α-Fe2O3 hematite").
2. Nhận xét về kích thước tinh thể (crystallite size) và microstrain dựa trên kết quả
   Williamson-Hall (nếu có) hoặc ước tính Scherrer.
3. Cảnh báo các đỉnh có thể là tạp chất hoặc pha vô định hình.
4. Đánh giá mức độ tin cậy (thấp/trung bình/cao) và đề xuất bước phân tích tiếp theo.

QUY TẮC NGHIÊM NGẶT:
- Tên pha: LUÔN viết tiếng Anh theo chuẩn khoáng vật học.
- Thuật ngữ khoa học (FWHM, chỉ số hkl, Bragg, Williamson-Hall): giữ tiếng Anh trong câu Việt.
- KHÔNG được bịa số ICDD card hay PDF entry nếu không chắc chắn. Nếu cite tham chiếu,
  đánh dấu "tentative" trừ khi 2θ khớp trong ±0.1°.
- Mức "cao" chỉ cho phép khi ≥ 5 đỉnh khớp một pha duy nhất trong ±0.2°.

Chỉ trả về JSON, không markdown fences."""


def system_prompt(locale: str) -> str:
    """Return system prompt based on locale."""
    if locale.lower().startswith("vi"):
        return XRD_SYSTEM_VI
    return XRD_SYSTEM_EN


USER_TEMPLATE = """Spectrum metadata:
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

Provide analysis as JSON with this schema:
{{
  "summary": "<3-sentence narrative in target language>",
  "phases": [
    {{
      "name": "<English phase name>",
      "confidence": "low|medium|high",
      "matched_peaks": <int>,
      "note": "<short narrative in target language>"
    }}
  ],
  "crystallite_size_nm": <number | null>,
  "microstrain": <number | null>,
  "warnings": ["<narrative in target language>"],
  "next_steps": ["<narrative in target language>"],
  "overall_confidence": "low|medium|high"
}}"""


def build_user_prompt(parsed: dict, metadata: dict) -> str:
    """Construct user message from parser output + Firestore metadata."""
    peaks = parsed.get("peaks", [])[:15]  # cap context
    peaks_table_rows = [
        f"  {i + 1:2d}. 2θ={p['two_theta']:7.3f}°  I={p['intensity']:9.1f}  "
        f"FWHM={p['fwhm']:.3f}°  I_rel={p['relative_intensity']:.1f}%"
        for i, p in enumerate(peaks)
    ]
    qs = parsed.get("quick_stats", {})
    return USER_TEMPLATE.format(
        spectrum_type=parsed.get("spectrum_type", "?"),
        instrument=metadata.get("instrument", "(not specified)"),
        sample_label=metadata.get("sampleLabel", "(unknown)"),
        wavelength=parsed.get("wavelength_angstrom", "?"),
        source=parsed.get("source", "?"),
        row_count=qs.get("rowCount", "?"),
        x_range=qs.get("xRange", "?"),
        peak_count=qs.get("peakCount", "?"),
        n_peaks_shown=len(peaks),
        peaks_table="\n".join(peaks_table_rows) if peaks_table_rows else "  (none)",
        scherrer_nm=parsed.get("scherrer_avg_nm", "n/a"),
        wh_json=parsed.get("williamson_hall") or "n/a (< 5 peaks)",
    )
