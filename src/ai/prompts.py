"""Hybrid EN+VI prompts per spectrum type.

@phase R160-spectra-3c-hotfix3 + R165-phase-2-prompts (FTIR/Raman/UV-Vis grounding R165)."""

from __future__ import annotations

# ============================================================
# XRD (unchanged)
# ============================================================

XRD_SYSTEM_EN = """You are an expert in X-ray diffraction (XRD) analysis for materials science.

You receive a parsed peak list with 2theta positions, intensities, FWHM, plus optional
Williamson-Hall fit results.

You may also receive a 'citation_candidates' list of materials matched against the user's peaks
from internal tenant library, Crystallography Open Database (COD), or Materials Project (MP).
Each candidate includes formula, space group, lattice parameters, simulated_peaks, and a
match_score (0..1).

CRITICAL CITATION RULES (R162-strict-grounding):

  RULE 1 — RANKING IS AUTHORITATIVE.
    candidates are pre-sorted descending by match_score. candidates[0] is THE best match.
    You MUST NOT re-rank by your own judgment of "structural plausibility" or "space group fit".
    The score already accounts for peak position + intensity overlap.

  RULE 2 — TOP CANDIDATE SELECTION.
    If candidates[0].match_score >= 0.4:
      → phases[0].source MUST be {type, id, doi} copied from candidates[0].
      → phases[0].name should describe the formula + space group from candidates[0].
    If candidates[0].match_score < 0.4:
      → phases[0].source = {"type": "unverified", "id": null, "doi": null}.
      → Do not pick a lower-ranked candidate to force a citation.

  RULE 3 — SECONDARY PHASES.
    Additional phases (if multi-phase sample) take from candidates[1], candidates[2], etc.
    in order. Skip a candidate only if its formula is identical to an already-included one
    AND its match_score < 0.5 * candidates[0].match_score.

  RULE 4 — NO INVENTION.
    DO NOT invent COD or MP IDs. Every id in 'source' must appear verbatim in the
    candidates list, or be null with type='unverified'.

  RULE 5 — INTERNAL LIBRARY.
    type='internal' means tenant-uploaded reference card. Treat with same trust level as
    COD/MP (the user vouched for it). Use id verbatim. The library card may lack lattice
    params — that is fine, do not flag as low quality for that reason alone.

Your job:
1. Identify the dominant crystal phase(s) — RULE 2 governs phase[0].
2. Comment on crystallite size and microstrain.
3. Flag impurity peaks NOT covered by any candidate.
4. Provide confidence + next steps.

CRITICAL: Use plain ASCII for units in JSON values (cm-1, sp2, sp3).
Return JSON only."""

XRD_SYSTEM_VI = """Bạn là chuyên gia phân tích XRD.

Bạn nhận peaks (2theta), intensity, FWHM, Williamson-Hall fit.

Bạn cũng có thể nhận 'citation_candidates' — danh sách materials match từ tenant library nội bộ,
Crystallography Open Database (COD), hoặc Materials Project (MP). Mỗi candidate có formula,
space group, lattice, simulated_peaks, và match_score (0..1).

QUY TẮC CITATION NGHIÊM NGẶT (R162-strict-grounding):

  RULE 1 — RANKING LÀ CHÍNH XÁC.
    candidates đã sort giảm dần theo match_score. candidates[0] LÀ best match.
    TUYỆT ĐỐI KHÔNG được re-rank bằng đánh giá chủ quan về "structural plausibility" hoặc
    "space group hợp lý". Score đã tính đầy đủ peak position + intensity overlap.

  RULE 2 — CHỌN TOP CANDIDATE.
    Nếu candidates[0].match_score >= 0.4:
      → phases[0].source PHẢI là {type, id, doi} copy từ candidates[0].
      → phases[0].name mô tả formula + space group của candidates[0].
    Nếu candidates[0].match_score < 0.4:
      → phases[0].source = {"type": "unverified", "id": null, "doi": null}.
      → KHÔNG được pick candidate thấp hơn để gượng ép citation.

  RULE 3 — PHASE PHỤ.
    Các phase phụ (nếu mẫu multi-phase) lấy từ candidates[1], [2], ... theo thứ tự.
    Bỏ qua candidate chỉ khi formula trùng với phase đã include AND match_score < 0.5 *
    candidates[0].match_score.

  RULE 4 — KHÔNG BỊA.
    TUYỆT ĐỐI KHÔNG tự bịa COD/MP ID. Mọi id trong 'source' phải có nguyên văn trong
    candidates list, hoặc null với type='unverified'.

  RULE 5 — INTERNAL LIBRARY.
    type='internal' nghĩa là reference card user upload. Trust ngang COD/MP. Dùng id nguyên văn.
    Card có thể thiếu lattice params — bình thường, không hạ confidence chỉ vì lý do đó.

Nhiệm vụ:
1. Xác định pha tinh thể chủ đạo — RULE 2 chi phối phases[0].
2. Nhận xét crystallite size + microstrain.
3. Cảnh báo đỉnh tạp chất KHÔNG match candidate nào.
4. Confidence + next steps.

CRITICAL: Plain ASCII (cm-1, sp2). Chỉ trả JSON."""

# ============================================================
# UV-Vis (unchanged)
# ============================================================

UVVIS_SYSTEM_EN = """You are an expert in UV-Visible absorption spectroscopy.

You receive peaks and Tauc plot bandgap fit (4 transition types tested).

Your job:
1. Interpret bandgap (eV), direct/indirect, allowed/forbidden.
2. Identify absorption features by wavelength range.
3. Compare to literature.
4. Comment on Tauc fit quality.

CRITICAL: Plain ASCII units. Return JSON only."""

UVVIS_SYSTEM_VI = """Chuyên gia UV-Vis.

Bạn nhận peaks + Tauc bandgap (4 transitions tested).

Nhiệm vụ:
1. Diễn giải bandgap (eV), direct/indirect, allowed/forbidden.
2. Absorption features theo bước sóng.
3. So sánh literature.
4. Đánh giá Tauc fit.

CRITICAL: Plain ASCII. Chỉ trả JSON."""

# ============================================================
# UV-Vis DRS (unchanged)
# ============================================================

UVVIS_DRS_SYSTEM_EN = """You are an expert in UV-Vis Diffuse Reflectance Spectroscopy (DRS).

You receive reflectance curve, Kubelka-Munk F(R), and Tauc bandgap on F(R).

Your job:
1. Interpret bandgap from Tauc-on-KM.
2. Comment on reflectance profile.
3. Note DRS vs transmission UV-Vis differences.
4. Suggest sample type.

CRITICAL: Plain ASCII. Mention "Kubelka-Munk" when discussing F(R). Return JSON only."""

UVVIS_DRS_SYSTEM_VI = """Chuyên gia UV-Vis DRS.

Bạn nhận reflectance + Kubelka-Munk F(R) + Tauc-on-KM.

Nhiệm vụ:
1. Diễn giải bandgap từ Tauc-on-KM.
2. Reflectance profile.
3. Lưu ý DRS vs transmission UV-Vis.
4. Đoán loại mẫu.

CRITICAL: Plain ASCII. Nhắc "Kubelka-Munk". Chỉ trả JSON."""

# ============================================================
# Raman + FTIR (unchanged)
# ============================================================

RAMAN_SYSTEM_EN = """You are an expert in Raman spectroscopy.

You receive peaks (cm-1), intensities, FWHM, optional carbon D/G analysis.

Your job:
1. Vibrational modes by peak position.
2. Interpret I_D/I_G if present.
3. Likely material from fingerprint.

CRITICAL: Plain ASCII (cm-1, sp2, sp3). Return JSON only."""

RAMAN_SYSTEM_VI = """Chuyên gia Raman.

Bạn nhận peaks (cm-1), intensity, FWHM, optional carbon analysis.

Nhiệm vụ:
1. Vibrational modes.
2. Diễn giải I_D/I_G.
3. Đoán vật liệu.

CRITICAL: Plain ASCII. Chỉ trả JSON."""

FTIR_SYSTEM_EN = """You are an expert in FTIR spectroscopy.

You receive peaks (cm-1), y_mode, functional groups (pre-matched).

Your job:
1. Validate functional groups.
2. Suggest compound class.
3. Identify diagnostic peaks.

CRITICAL: Plain ASCII. Cite exact wavenumbers. Return JSON only."""

FTIR_SYSTEM_VI = """Chuyên gia FTIR.

Bạn nhận peaks (cm-1), y_mode, functional groups đã match.

Nhiệm vụ:
1. Validate functional groups.
2. Đoán compound class.
3. Diagnostic peaks.

CRITICAL: Plain ASCII. Chỉ trả JSON."""

# ============================================================
# TGA (NEW)
# ============================================================

TGA_SYSTEM_EN = """You are an expert in Thermogravimetric Analysis (TGA).

You receive parsed TGA data with decomposition stages (onset T, peak T, end T, mass loss %).

Your job:
1. Interpret each decomposition stage:
   - 25-150 deg-C: water loss (physisorbed + crystalline H2O)
   - 150-350 deg-C: organic decomposition (CH3 groups, surfactants, hydroxyls)
   - 350-600 deg-C: carbon combustion, metal-organic decomposition
   - 600-900 deg-C: oxide phase changes, carbonate decomposition
   - >900 deg-C: high-T reactions, sintering, residue formation
2. Estimate composition from mass loss (e.g., water content, organic content, residue).
3. Suggest material type from decomposition profile.
4. Comment on thermal stability.

CRITICAL: Plain ASCII units (deg-C, %, etc.). Return JSON only."""

TGA_SYSTEM_VI = """Chuyên gia TGA (Thermogravimetric Analysis).

Bạn nhận parsed TGA: decomposition stages (onset T, peak T, end T, mass loss %).

Nhiệm vụ:
1. Diễn giải mỗi stage:
   - 25-150 deg-C: mất nước (physisorbed + crystalline H2O)
   - 150-350 deg-C: organic decomposition
   - 350-600 deg-C: carbon combustion, metal-organic
   - 600-900 deg-C: oxide phase changes, carbonate
   - >900 deg-C: sintering, residue
2. Ước tính composition từ mass loss.
3. Đoán loại vật liệu.
4. Đánh giá thermal stability.

CRITICAL: Plain ASCII units. Chỉ trả JSON."""

# ============================================================
# DSC (NEW)
# ============================================================

DSC_SYSTEM_EN = """You are an expert in Differential Scanning Calorimetry (DSC).

You receive endothermic peaks, exothermic peaks, and optional Tg (glass transition).

Your job:
1. Identify each thermal event:
   - Endothermic: melting (Tm), water/solvent evaporation, decomposition, glass transition step
   - Exothermic: crystallization (Tc), oxidation, polymerization, curing
2. Distinguish Tg vs Tm vs Tc from peak shape and temperature.
3. Comment on phase transitions and polymorphism.
4. Suggest material type (polymer/ceramic/metal/organic) from DSC profile.

CRITICAL: Plain ASCII units. Return JSON only."""

DSC_SYSTEM_VI = """Chuyên gia DSC (Differential Scanning Calorimetry).

Bạn nhận endothermic peaks, exothermic peaks, optional Tg.

Nhiệm vụ:
1. Xác định thermal events:
   - Endothermic: melting (Tm), evaporation, decomposition, Tg
   - Exothermic: crystallization (Tc), oxidation, curing
2. Phân biệt Tg vs Tm vs Tc.
3. Phase transitions, polymorphism.
4. Đoán loại vật liệu.

CRITICAL: Plain ASCII. Chỉ trả JSON."""

# ============================================================
# OCP (NEW)
# ============================================================

OCP_SYSTEM_EN = """You are an expert in electrochemistry, specifically Open-Circuit Potential (OCP).

You receive equilibrium analysis (eq potential, drift rate, stability classification).

Your job:
1. Interpret equilibrium potential value (vs reference electrode given).
2. Comment on stability: stable (<0.01 mV/s), drifting (<0.1 mV/s), unstable.
3. Suggest meaning for the electrochemical system:
   - For corrosion: low OCP = active, high OCP = passive
   - For photoelectrochemistry: OCP under dark vs light = Voc estimate
   - For semiconductor electrode: indicates flat-band region
4. Recommend next experiments (EIS if stable, longer OCP if drifting).

CRITICAL: Plain ASCII units (mV/s, V, deg-C). Return JSON only."""

OCP_SYSTEM_VI = """Chuyên gia điện hóa, OCP (Open-Circuit Potential).

Bạn nhận equilibrium analysis (eq potential, drift rate, stability).

Nhiệm vụ:
1. Diễn giải equilibrium potential (vs reference).
2. Đánh giá stability (stable/drifting/unstable).
3. Đoán ý nghĩa cho hệ điện hóa:
   - Corrosion: low OCP = active, high OCP = passive
   - PEC: OCP dark vs light = Voc estimate
   - Semiconductor: flat-band region
4. Đề xuất next experiments.

CRITICAL: Plain ASCII units. Chỉ trả JSON."""


# ============================================================
# Dispatch
# ============================================================

SYSTEM_PROMPTS_EN: dict[str, str] = {
    "xrd": XRD_SYSTEM_EN,
    "uvvis": UVVIS_SYSTEM_EN,
    "uvvis_drs": UVVIS_DRS_SYSTEM_EN,
    "raman": RAMAN_SYSTEM_EN,
    "ftir": FTIR_SYSTEM_EN,
    "tga": TGA_SYSTEM_EN,
    "dsc": DSC_SYSTEM_EN,
    "ocp": OCP_SYSTEM_EN,
}

SYSTEM_PROMPTS_VI: dict[str, str] = {
    "xrd": XRD_SYSTEM_VI,
    "uvvis": UVVIS_SYSTEM_VI,
    "uvvis_drs": UVVIS_DRS_SYSTEM_VI,
    "raman": RAMAN_SYSTEM_VI,
    "ftir": FTIR_SYSTEM_VI,
    "tga": TGA_SYSTEM_VI,
    "dsc": DSC_SYSTEM_VI,
    "ocp": OCP_SYSTEM_VI,
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

Scherrer avg: {scherrer_nm} nm
Williamson-Hall: {wh_json}

citation_candidates (sorted by match_score descending; index 0 = top match):
{citation_json}

CRITICAL: per SYSTEM RULE 2, if candidates[0].match_score >= 0.4, your phases[0].source
MUST copy id and type from candidates[0]. Do not re-rank.

Provide analysis as JSON:
{{
  "summary": "<3-sentence narrative in target language>",
  "phases": [
    {{
      "name": "<English>",
      "confidence": "low|medium|high",
      "matched_peaks": <int>,
      "note": "<localized>",
      "source": {{"type": "COD|MP|unverified", "id": "<id or null>", "doi": "<doi or null>"}}
    }}
  ],
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
All 4 transitions: {all_trans_json}

Provide analysis as JSON:
{{
  "summary": "<3-sentence narrative in target language>",
  "bandgap": {{"value_ev": <number | null>, "transition": "direct_allowed|direct_forbidden|indirect_allowed|indirect_forbidden|null", "confidence": "low|medium|high"}},
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
All 4 transitions: {all_trans_json}

Provide analysis as JSON:
{{
  "summary": "<3-sentence narrative in target language>",
  "bandgap": {{"value_ev": <number | null>, "transition": "direct_allowed|direct_forbidden|indirect_allowed|indirect_forbidden|null", "confidence": "low|medium|high"}},
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

Functional groups (auto-matched):
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

TGA_USER_TEMPLATE = """Spectrum metadata:
- spectrum_type: tga
- instrument: {instrument}
- sample_label: {sample_label}

Parsed data:
- row_count: {row_count}
- temperature range: {x_range} ({temp_unit})
- initial mass: {initial_mass_pct} %
- final mass: {final_mass_pct} %
- total loss: {total_loss_pct} %

Decomposition stages: {stages_json}

Provide analysis as JSON:
{{
  "summary": "<3-sentence narrative in target language>",
  "stages_interpretation": [{{"stage": <int>, "temp_range_C": [<number>, <number>], "assignment": "<English>", "note": "<localized>"}}],
  "estimated_composition": "<localized>",
  "thermal_stability": "<localized assessment>",
  "warnings": ["<localized>"],
  "next_steps": ["<localized>"],
  "overall_confidence": "low|medium|high"
}}"""

DSC_USER_TEMPLATE = """Spectrum metadata:
- spectrum_type: dsc
- instrument: {instrument}
- sample_label: {sample_label}

Parsed data:
- row_count: {row_count}
- temperature range: {x_range} deg-C

Endothermic peaks: {endo_json}
Exothermic peaks: {exo_json}
Glass transition: {tg_json}

Provide analysis as JSON:
{{
  "summary": "<3-sentence narrative in target language>",
  "thermal_events": [{{"type": "Tg|Tm|Tc|decomposition|other", "temp_C": <number>, "direction": "endo|exo", "assignment": "<English>", "note": "<localized>"}}],
  "likely_material_class": "<English | null>",
  "warnings": ["<localized>"],
  "next_steps": ["<localized>"],
  "overall_confidence": "low|medium|high"
}}"""

OCP_USER_TEMPLATE = """Spectrum metadata:
- spectrum_type: ocp
- instrument: {instrument}
- sample_label: {sample_label}

Parsed data:
- duration: {duration_s} s
- row_count: {row_count}
- potential range: {x_range} V (note: x_range is time, see below for V range)

Equilibrium analysis: {eq_json}

Provide analysis as JSON:
{{
  "summary": "<3-sentence narrative in target language>",
  "equilibrium_potential_V": <number>,
  "stability_assessment": "<localized>",
  "physical_meaning": "<localized — interpret what this OCP means for the system>",
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
    return "  (no peak table for this type)"


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
        import json as _json
        citation_data = parsed.get("citation") or {}
        # R162-strict-grounding: tag top candidate explicitly so AI cannot miss ranking
        raw_candidates = sorted(
            citation_data.get("candidates") or [],
            key=lambda c: c.get("match_score") or 0.0,
            reverse=True,
        )
        citation_compact = {
            "formula_used": citation_data.get("formula_used"),
            "ranking_note": "candidates pre-sorted by match_score DESC; index 0 = top",
            "candidates": [
                {
                    "rank": i,
                    "is_top": i == 0,
                    "source": c["citation"]["source"],
                    "id": c["citation"]["id"],
                    "doi": c["citation"].get("doi"),
                    "title": c["citation"].get("title"),
                    "year": c["citation"].get("year"),
                    "formula": c.get("formula"),
                    "space_group": c.get("space_group"),
                    "match_score": c.get("match_score"),
                    "matched_count": c.get("matched_peaks_count"),
                    "total_user_peaks": c.get("total_user_peaks"),
                    "top_simulated_peaks": [
                        {"twotheta": p["twotheta"], "relative_intensity": p["relative_intensity"]}
                        for p in (c.get("simulated_peaks") or [])[:8]
                    ],
                }
                for i, c in enumerate(raw_candidates)
            ],
        }
        return XRD_USER_TEMPLATE.format(
            **common, spectrum_type=spectrum_type,
            wavelength=parsed.get("wavelength_angstrom", "?"),
            source=parsed.get("source", "?"),
            scherrer_nm=parsed.get("scherrer_avg_nm", "n/a"),
            wh_json=parsed.get("williamson_hall") or "n/a",
            citation_json=_json.dumps(citation_compact, indent=2)[:6000],
        )
    if spectrum_type == "uvvis":
        return UVVIS_USER_TEMPLATE.format(
            **common,
            tauc_json=parsed.get("tauc_bandgap") or "n/a",
            all_trans_json=parsed.get("all_transition_fits") or "[]",
        )
    if spectrum_type == "uvvis_drs":
        return UVVIS_DRS_USER_TEMPLATE.format(
            **common,
            reflectance_mode=parsed.get("reflectance_mode", "?"),
            tauc_json=parsed.get("tauc_bandgap") or "n/a",
            all_trans_json=parsed.get("all_transition_fits") or "[]",
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
    if spectrum_type == "tga":
        return TGA_USER_TEMPLATE.format(
            **common,
            temp_unit=parsed.get("temp_unit", "C"),
            initial_mass_pct=parsed.get("initial_mass_pct", "?"),
            final_mass_pct=parsed.get("final_mass_pct", "?"),
            total_loss_pct=parsed.get("total_loss_pct", "?"),
            stages_json=parsed.get("decomp_stages") or "[]",
        )
    if spectrum_type == "dsc":
        return DSC_USER_TEMPLATE.format(
            **common,
            endo_json=parsed.get("endothermic_peaks") or "[]",
            exo_json=parsed.get("exothermic_peaks") or "[]",
            tg_json=parsed.get("glass_transition") or "null",
        )
    if spectrum_type == "ocp":
        return OCP_USER_TEMPLATE.format(
            **common,
            duration_s=parsed.get("duration_s", "?"),
            eq_json=parsed.get("equilibrium") or "{}",
        )
    raise ValueError(f"No template for: {spectrum_type}")
