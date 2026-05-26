# Tafel Analysis (HER/OER Electrocatalytic Kinetics)

Module: `src/parsers/tafel.py` · Phase: R260 · Cluster: electrochemistry

## Purpose

Derive **kinetic** parameters from a polarization curve that the activity
benchmark (overpotential at 10 mA/cm², see `lsv-analysis.md`) does not provide:
the Tafel slope, the exchange current density `j0`, the transfer coefficient
`alpha`, and a rate-determining-step hint for the hydrogen evolution reaction
(HER). This complements — does not replace — the LSV figure of merit.

## Methods and formulae

### Tafel equation

For an electrode reaction far from equilibrium (|η| ≳ 50–100 mV), the
overpotential and current density follow the Tafel relation:

```
eta = a + b * log10(|j|)
```

where `b` is the **Tafel slope** (V/dec) and `a` the intercept. The slope is
related to the transfer coefficient by the Butler–Volmer kinetics:

```
b = 2.303 * R * T / (alpha * F)
```

At T = 298.15 K, `2.303·R·T/F = 0.0592 V`, so:

```
alpha = 0.0592 / |b|        (b in V/dec, 25 C, 1-electron transfer)
```

A slope of ~118–120 mV/dec corresponds to `alpha ≈ 0.5`, the signature of a
rate-limiting one-electron (Volmer) step.

### Exchange current density

`j0` is the current density at zero overpotential, obtained by extrapolating the
Tafel line to η = 0:

```
0 = a + b * log10(j0)   ->   log10(j0) = -a / b   ->   j0 = 10^(-a/b)
```

Higher `j0` means a more intrinsically active catalyst. `j0` is **sensitive** to
the chosen linear region and to uncompensated resistance (iR), so it is reported
with the fit R² and flagged when R² < 0.98 or iR correction is unknown.

### HER mechanism by Tafel slope (approximate, Bard §15.2.2)

| Tafel slope (mV/dec) | Rate-determining step |
|---|---|
| ~120 | Volmer: H⁺ + e⁻ → H_ads (α ≈ 0.5) |
| ~40  | Heyrovský: H_ads + H⁺ + e⁻ → H₂ |
| ~30  | Tafel recombination: 2 H_ads → H₂ |

These are textbook reference values; real systems vary, so the output is a
*hint*, not an assignment. **OER** is a 4-electron multistep reaction; its slope
does not map to a single step and is reported as such.

### Implementation notes

- Input is read through `_tabular.load_spectrum` (vendor-header strip + EU
  decimal + column detection), so CorrWare/ZPlot/Gamry/Bio-Logic exports work.
- Potential is converted to the RHE scale (`E_RHE = E_meas + offset + 0.059·pH`)
  using the reference-electrode offsets shared with `lsv.py`. Overpotential:
  HER `η = −E_RHE`, OER `η = E_RHE − 1.23`.
- The linear region is selected by sliding a window (≥5 points, ≥ n/3 wide) over
  the kinetic branch (|j| > 1 µA, correct overpotential sign) and keeping the
  fit with the highest R².
- `j0`/`alpha` are withheld unless reaction (her/oer), reference electrode and pH
  are provided. Without electrode area, `j0` is in raw current units and flagged.

## Edge cases and guards

- No clear linear region (noisy/short/mass-transport-dominated) → no fabricated
  fit; a note is returned.
- R² < 0.98 → parameters flagged unreliable; verify kinetic region + iR.
- Tafel parameters are emphasised as **mechanism indicators, not the activity
  benchmark**; the primary figure of merit remains η at 10 mA/cm² (LSV).

## References

- Bard, A. J.; Faulkner, L. R.; White, H. S. *Electrochemical Methods:
  Fundamentals and Applications*, 3rd ed.; Wiley, 2022. §15.2.2 (Tafel plot
  analysis of HER kinetics). ISBN 9781119334064.
- McCrory, C. C. L. et al. *J. Am. Chem. Soc.* **2013**, 135, 16977.
  DOI: 10.1021/ja407115p. (Ref 69: rationale for not using Tafel as the activity
  metric — multistep, system-specific.)
- McCrory, C. C. L. et al. *J. Am. Chem. Soc.* **2015**, 137, 4347.
  DOI: 10.1021/ja510442p.
