# LSV (Linear Sweep Voltammetry) — HER/OER Electrocatalysis

Scientific methods used by `src/parsers/lsv.py`. Implementation: `parse_lsv()`.

Computes the standard activity benchmarks for water-splitting electrocatalysts
from a polarization curve (potential, current).

## 1. Current density

j = I / A (mA/cm2, geometric area). Without the electrode area, current is kept
raw and benchmarks (which are defined per cm2) are not computed.

## 2. RHE conversion (required for overpotential)

```
E_RHE = E_measured + E_ref_offset + 0.059 * pH    (25 C)
```

Reference offsets (V): Ag/AgCl sat. KCl 0.197, Ag/AgCl 3M 0.210, SCE 0.241,
Hg/HgO 1M 0.140, RHE/SHE 0. **Without the reference electrode and pH, the
overpotential is not computed** (no silent assumption).

## 3. Overpotential

```
OER:  eta = E_RHE - 1.23 V      (O2/H2O equilibrium)
HER:  eta = 0 - E_RHE           (H+/H2 equilibrium; cathodic, current negative)
```

- **overpotential_at_10mA_cm2_V** — the field of merit: the overpotential to
  reach 10 mA/cm2 geometric, ~the current of a 10%-efficient solar water-
  splitting device. Lower = more active.
- **onset_overpotential_at_1mA_cm2_V** — onset defined at 1 mA/cm2.

Both are linearly interpolated between samples.

## 4. Tafel slope

From the high-overpotential (Butler-Volmer) approximation, eta = a + b*log10|j|
is linear. The parser scans for the most linear window (max R^2, span >= 0.3
decade) and reports b in mV/dec. Reference values: HER on Pt ~30 mV/dec.

## 5. Caveats emitted as notes

- **Not iR-corrected**: overpotential and Tafel slope are overestimated; iR
  compensation can change benchmarks by tens of mV (and the Tafel slope).
- **Reference/pH unknown** -> eta not computed.
- **Area unknown** -> raw current, no per-cm2 benchmarks.
- **Reaction unknown** -> provide her/oer.
- Geometric activity can be dominated by ECSA, not intrinsic activity; the
  10 mA/cm2 onset can be distorted by large pseudocapacitive current.

## References

- McCrory et al., J. Am. Chem. Soc. 135 (2013) 16977 — benchmarking HER/OER,
  10 mA/cm2 figure of merit.
- Anantharaj et al.; RSC Adv. 13 (2023) — water-splitting electrochemical
  parameters (Tafel from Butler-Volmer).
- Bard, Faulkner & White, Electrochemical Methods, 3rd ed. (2022) — electrode
  kinetics, Tafel/Butler-Volmer.
- iR compensation: Nat./ACS Energy Lett. 8 (2023) recommendations.
