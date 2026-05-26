# CV (Cyclic Voltammetry) Analysis

Scientific methods used by `src/parsers/cv.py`. Implementation: `parse_cv()`.

## 1. Sweep splitting

The voltammogram is split at the most-positive potential (switching potential):
the anodic sweep is E increasing, the cathodic sweep is E decreasing. Peaks are
located per branch (anodic = maximum positive current, cathodic = maximum
negative current).

## 2. Redox-couple descriptors

- **Epa, Epc** — anodic / cathodic peak potentials; **ipa, ipc** — peak currents.
- **dEp = Epa - Epc** — peak separation. For an ideal Nernstian (reversible)
  n-electron couple, dEp ~ 59/n mV at 25 C. Solid electrodes typically show
  larger values (70-80 mV) even when reversible.
- **E0' = (Epa + Epc)/2** — formal potential (when both peaks are present).
- **peak_current_ratio = |ipa/ipc|** — ~1 for a reversible couple.

## 3. Reversibility classification (provisional)

| dEp (mV) and ratio | class |
|---|---|
| <= ~80 and 0.8-1.25 | reversible-like |
| <= ~200 | quasi-reversible |
| > 200 | irreversible-like |

**One scan rate is not enough to conclude reversibility.** The diagnostic test
is the scan-rate dependence: dEp constant with v (reversible) vs increasing with
v (quasi-reversible); and ip vs sqrt(v) linear (diffusion-controlled, Randles-
Sevcik) vs ip vs v linear (surface-confined/adsorbed). The parser flags this.

## 4. Not computed from a single CV

- **Randles-Sevcik** (ip = 2.69e5 n^1.5 A D^0.5 C v^0.5 at 25 C) and **ECSA /
  double-layer capacitance** require a scan-rate series, not one voltammogram;
  the parser notes this rather than fabricating a value.

## 5. Sign convention (IUPAC)

Positive scan direction + positive current = oxidation (anodic); negative scan +
negative current = reduction (cathodic).

## References

- Elgrishi et al., J. Chem. Educ. 95 (2018) 197 - practical CV guide
  (dEp 57 mV, fwhm 59 mV for reversible 1e couple).
- Randles-Sevcik equation (peak current vs scan rate).
- Bard, Faulkner & White, Electrochemical Methods, 3rd ed. (2022).
- Morales & Risch, J. Phys. Energy 3 (2021) - reliable CV for Cdl/ECSA.
