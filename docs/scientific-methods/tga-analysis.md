# TGA (Thermogravimetric Analysis)

Scientific methods used by `src/parsers/tga.py`. Implementation: `parse_tga()`.

## 1. Curve preparation

Temperature unit auto-detected (K if min > 200 and max > 300, else °C). Mass
auto-detected as percent (0–110) or absolute, normalised to initial mass for %.
DTG = −dm/dT computed on a Savitzky–Golay-smoothed mass curve.

## 2. Decomposition stages

DTG peaks (`scipy.signal.find_peaks`, prominence ≥ 5% of DTG span) mark stages.
Per stage:

- **peak_T** — DTG maximum = temperature of maximum decomposition rate (T_d).
  Mathematically well-defined.
- **deviation_onset_T** — first deflection from baseline (ASTM E2550 style);
  depends on display magnification, so it is the *least* reproducible.
- **extrapolated_onset_T** — see §3 (the reported standard).
- **end_T**, **mass_loss_pct**, **dtg_max**.

## 3. Extrapolated onset temperature (ISO 11358-1)

> "The point of intersection of the starting-mass baseline and the tangent to
> the TGA curve at the point of maximum gradient." — ISO 11358-1 / DIN 51006.

```
tangent at DTG peak:  m(T) = m_peak + slope·(T − T_peak),   slope = dm/dT = −DTG_peak
pre-step baseline:    m = mean(mass over the flat region just before the step)
extrapolated onset:   T_onset = T_peak + (baseline − m_peak) / slope
```

This is distinct from, and higher than, the deviation onset, and lower than the
DTG peak. (Example from literature: 219 °C deviation / 299 °C extrapolated /
315 °C DTG for one step — all three differ.) **Implementation:** `_extrapolated_onset()`.

## 4. Thermal-stability indices

Temperature at which cumulative mass loss first reaches a threshold, linearly
interpolated between samples:

- **T5%**, **T10%** — common thermal-stability descriptors.
- **T50%** — half-decomposition temperature.

**Implementation:** `_temperature_at_loss()`.

## 5. Char yield

Residual mass at the end of the run (`char_yield_pct` = `final_mass_pct`).
Important for polymers/composites and flame-retardant evaluation.

## 6. Edge cases

- DTG slope ≈ 0 at a peak → extrapolated onset None (avoids divide-by-zero).
- Extrapolated onset outside the data range or above the peak → None (rejected).
- Onset/end search is unbounded (walks until DTG drops below 10% of the peak),
  so broad single steps are bounded correctly.

## References

- ISO 11358-1 — Plastics, Thermogravimetry (TG), Part 1: General principles.
- DIN 51006 — Thermal analysis (TA), Thermogravimetry (TG).
- ASTM E2550 — Standard Test Method for Thermal Stability by Thermogravimetry.
- ASTM E1131 — Standard Test Method for Compositional Analysis by Thermogravimetry.
