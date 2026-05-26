# Raman Spectroscopy Analysis

Scientific methods used by `src/parsers/raman.py`. Implementation: `parse_raman()`.

## 1. Peak detection

Savitzky–Golay smoothing (window 11, polyorder 3) then `scipy.signal.find_peaks`
with prominence = 3% of signal span, min distance 5 points, min width 2.
Reported per peak: shift (cm⁻¹), baseline-subtracted height, FWHM (cm⁻¹),
relative intensity (%).

## 2. Crystallite size La (in-plane, sp² carbon)

**This is excitation-wavelength dependent — La is NOT computed without λ.**

The integrated-intensity ratio I_D/I_G is inversely proportional to the fourth
power of the laser energy, so a single proportionality constant is wrong across
excitation lines.

- **Cançado (2006)** general equation (used here):
  `La (nm) = 2.4×10⁻¹⁰ · λ⁴ · (I_D/I_G)⁻¹`, λ in nm, **integrated** D/G ratio.
  DOI: 10.1063/1.2196057 (Appl. Phys. Lett. 88, 163106).
- **Tuinstra–Koenig (1970)**: `La = 4.4 · (I_D/I_G)⁻¹` — only valid at 514.5 nm.
  Emitted as a cross-check note only when λ ≈ 514.5 nm.

If λ is unknown, La is `None` and a note explains why. No silent 514 nm default.

**Implementation:** `_crystallite_size_la()`, `_band_area()` (trapezoid over
center ± 2·FWHM, baseline-subtracted).

## 3. D/G ratio — height vs area

Two conventions, both reported (they are not interchangeable):
- `id_ig_ratio_height` — peak heights (Tuinstra–Koenig convention).
- `id_ig_ratio_area` — integrated areas (Cançado; used for La).
Reference: Pimenta et al.; Mallet-Ladeira et al. 2014 (height vs area caveat).

## 4. Graphene layer hint (2D band)

From the 2D band (~2690 cm⁻¹) shape and I_2D/I_G:
- single-layer: sharp 2D (FWHM ≤ 40 cm⁻¹) and I_2D/I_G > 2 (Ferrari 2006).
- few-layer: I_2D/I_G > 1. Otherwise multi-layer / graphite.

## 5. TMD layer-count (MoS₂ / WS₂)

E²g (in-plane) and A1g (out-of-plane) separation Δ grows with layer number.
- **MoS₂** (Lee 2010): Δ ≈ 18 (1L), ≈ 21–22 (2L), ≈ 25 (bulk).
  Modes ~383 (E²g) / ~408 (A1g) cm⁻¹.
- **WS₂** (Berkdemir 2013, Nature Sci. Rep. srep01755): ~351 (E²g/2LA(M)) /
  ~418 (A1g) cm⁻¹; A1g frequency decreases monotonically with layer count.

**Implementation:** `_tmd_analysis()`.

## 6. Band assignment table

`RAMAN_BANDS` — peaks annotated only when they fall inside a curated window.
No inference beyond the table. Sources per entry:

| Window (cm⁻¹) | Assignment | Material | Reference |
|---|---|---|---|
| 1300–1380 | D band | carbon | Ferrari & Robertson 2000 |
| 1560–1620 | G band | carbon | Ferrari & Robertson 2000 |
| 2650–2750 | 2D band | carbon | Ferrari 2006 |
| 375–387 | E²g | MoS₂ | Lee 2010 |
| 403–412 | A1g | MoS₂ | Lee 2010 |
| 345–360 | E²g / 2LA(M) | WS₂ | Berkdemir 2013 |
| 413–423 | A1g | WS₂ | Berkdemir 2013 |
| 800–820 | O–W–O stretch (~807) | WO₃ | Daniel 1987 (m-WO₃) |
| 700–730 | O–W–O stretch (~715) | WO₃ | Daniel 1987 (m-WO₃) |
| 265–285 | O–W–O bend (~273) | WO₃ | Daniel 1987 (m-WO₃) |
| 400–700 | Metal–O lattice | oxide | general (low specificity) |

## 7. Edge cases

- λ unknown → La None + note (never assume 514 nm).
- I_G = 0 → ratios None/0, La None.
- FWHM unresolved → area window falls back to ±20 cm⁻¹.
- Non-carbon spectrum → `carbon_analysis` is None (no spurious D/G).

## References

- Tuinstra & Koenig, J. Chem. Phys. 53, 1126 (1970).
- Ferrari & Robertson, Phys. Rev. B 61, 14095 (2000).
- Cançado et al., Appl. Phys. Lett. 88, 163106 (2006), DOI 10.1063/1.2196057.
- Ferrari et al., Phys. Rev. Lett. 97, 187401 (2006) — graphene 2D.
- Lee et al., ACS Nano 4, 2695 (2010) — MoS₂ layer dependence.
- Berkdemir et al., Sci. Rep. 3, 1755 (2013) — WS₂ identification.
