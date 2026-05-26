# FTIR (Fourier-Transform Infrared) Analysis

Scientific methods used by `src/parsers/ftir.py`. Implementation: `parse_ftir()`.

## 1. Input handling

- **Vendor headers** stripped: PerkinElmer Spectrum ASCII ("PE IR" + "#DATA"),
  JCAMP-DX (IUPAC ##LABEL=value, data after ##XYDATA=).
- **Y mode** auto-detected: transmittance (max 20–110), absorbance (max < 5).
- Transmittance converted to absorbance: A = −log10(%T / 100).
- Descending-x files (PerkinElmer ASC, 4000→400) sorted ascending so FWHM > 0.

## 2. Sampling mode and ATR correction

The sampling mode (`transmission` / `atr`) changes how intensities compare.

In ATR the evanescent-wave penetration depth dp is inversely proportional to
wavenumber, so **ATR over-weights low wavenumbers and under-weights high ones**
relative to transmission.

**Simple ATR correction** (first approximation):

```
A_corrected(wavenumber) = A_atr(wavenumber) · (wavenumber / 1000)
```

This compensates the penetration-depth scaling only. It does **not** remove the
anomalous-dispersion red-shift / band-shape change near strong bands — that
requires an advanced correction using the sample refractive index and a
Kramers-Kronig transform. Applied only when mode = "atr".

When mode is unknown, a note warns that low-wavenumber bands may be
over-weighted (in case the data is ATR).

References: Anton Paar ATR wiki; ScienceDirect ATR overview; Specac (dp theory).

## 3. Baseline + peak detection

Savitzky–Golay smoothing, then a straight baseline between the spectrum ends is
subtracted before peak heights are measured (reduces scattering/ATR drift bias).
Peaks via `scipy.signal.find_peaks` (prominence ≥ 3% of span). Each peak:
wavenumber (cm⁻¹), baseline-subtracted absorbance, FWHM.

## 4. Functional-group assignment

`FUNCTIONAL_GROUPS` — a peak is annotated when it falls inside a characteristic
window (O-H, C-H, C=O, C=C, C-O, metal-oxide, etc.). No inference beyond the
table.

## 5. Atmospheric-band flagging

Peaks inside known atmospheric-interference windows are flagged as possible
artefacts (to verify against the background scan):

| Window (cm⁻¹) | Species | Note |
|---|---|---|
| 2300–2380 | CO₂ | asymmetric stretch (~2349) |
| 660–670 | CO₂ | bend (~667) |
| 3500–3950 | H₂O | vapour stretch/rotational |
| 1300–1680 | H₂O | vapour bending (may overlap real bands) |

## 6. Edge cases

- mode unknown → no correction + note.
- atmospheric flags are advisory (esp. H₂O bending overlaps real bands).
- y mode "unknown" (max 5–20) → treated as absorbance without conversion.

## References

- ISO/standards on ATR sampling; Anton Paar "Attenuated total reflectance".
- Mayerhöfer et al. — limits of simple ATR correction (anomalous dispersion).
- Specac — depth of penetration for ATR measurements.
