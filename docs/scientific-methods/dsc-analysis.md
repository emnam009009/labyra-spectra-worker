# DSC (Differential Scanning Calorimetry) Analysis

Scientific methods used by `src/parsers/dsc.py`. Implementation: `parse_dsc()`.

## 1. Peak detection

Savitzky–Golay smoothing then `scipy.signal.find_peaks` in both directions:
exothermic (positive heat flow) and endothermic (negative). Each peak reports
peak temperature (°C), heat flow at apex, FWHM, direction, and enthalpy ΔH.

## 2. Peak enthalpy ΔH

**Heating-rate dependent — ΔH is NOT computed without β.**

Heat flow is recorded against temperature, but enthalpy is the time-integral of
power. The heating rate β converts the temperature axis to time:

```
ΔH = (1/β) · ∫ (HF − baseline) dT,   β in °C/s
```

- **Baseline**: linear interpolation between the two peak feet (located where the
  baseline-deviation signal returns to ≈5% of the apex). ΔH is the area above
  this baseline. Reference: Kong & Hay, Eur. Polym. J. 39 (2003) 1721.
- **Units**:
  - heat flow in **W/g** → ΔH directly in **J/g**.
  - heat flow in **mW** → ΔH in **mJ**; divide by sample mass (g) for **J/g**
    (so sampleMass in mg is required for a specific enthalpy).
- Without β, ΔH is `None` and a note explains why. No silent default β.

**Implementation:** `_integrate_enthalpy()`, `_peak_bounds()`.

## 3. Polymer crystallinity

```
Xc (%) = (ΔH_melt − ΔH_cold-cryst) / ΔH°f · 100
```

where ΔH°f is the heat of fusion of the 100%-crystalline polymer. The
cold-crystallization exotherm is subtracted because that fraction crystallized
in the instrument, not in the original sample (Kong & Hay 2003).

ΔH°f reference table (`REFERENCE_ENTHALPY_100`, J/g), peer-reviewed:

| Polymer | ΔH°f (J/g) | Source |
|---|---|---|
| PE / HDPE / LDPE | 293 | Mirabella & Bafna 2002 (J. Polym. Sci. B 40, 1637) |
| PP | 207 | isotactic PP, common literature |
| PET | 140.1 | poly(ethylene terephthalate) |
| PA6 | 230 | nylon-6 |
| PA66 | 196 | nylon-6,6 |
| PLA | 93 | poly(lactic acid) |
| PEEK | 130 | poly(ether ether ketone) |

Computed only when the caller declares a polymer present in the table.

## 4. Glass transition (Tg)

Inflection point = location of maximum |d²(HF)/dT²| in the low-temperature
baseline region. Reported with method label "inflection (ISO 11357-2)" and an
approximate ΔCp magnitude. ISO 11357-2 also defines onset and midpoint methods;
this parser reports the inflection convention.

## 5. Edge cases

- β unknown → ΔH None + note (never assume a rate).
- mW heat flow + unknown mass → ΔH in mJ + note (not J/g).
- polymer not declared / not in table → crystallinity None.
- Xc clamped to [0, 100].

## References

- ISO 11357-1/-2/-3 — Plastics, DSC.
- Kong, Y. & Hay, J.N., Eur. Polym. J. 39 (2003) 1721 — enthalpy of fusion &
  degree of crystallinity (cold-crystallization correction).
- Mirabella, F.M. & Bafna, A., J. Polym. Sci. B 40 (2002) 1637 — ΔH°f of PE.
