# EIS (Electrochemical Impedance Spectroscopy) Analysis

Scientific methods used by `src/parsers/eis.py`. Implementation: `parse_eis()`.

Two-tier analysis: a robust model-free readout that always runs, and an optional
equivalent-circuit fit seeded from it.

## 1. Input

2–4 columns. Rectangular `(freq, Z', Z'')` is the default; pass
`data_format="polar"` for `(freq, |Z|, phase_deg)`, which is converted by
Z' = |Z|·cos(phase), Z'' = |Z|·sin(phase).

The Z'' sign convention varies by instrument. The parser normalises so the
capacitive arc is negative (Nyquist plots −Z'' upward); a flipped sign is
detected and corrected automatically. **Auto-detection of polar vs rectangular
is intentionally not attempted** — Z'' magnitudes overlap the degree range and
would misclassify, so the format is explicit.

## 2. Tier 1 — model-free readout (always runs)

Read directly from the Nyquist data; never diverges.

- **Rs** (solution/series resistance) = Z' at the highest frequency
  (high-frequency real-axis intercept).
- **Rct** (charge-transfer resistance) = semicircle diameter. Taken as the real
  value where the arc closes after its apex (local minimum of −Z'' past the
  apex), minus Rs — this excludes any low-frequency Warburg tail.
- **Cdl** (double-layer capacitance) from the apex frequency:
  `omega_max = 1/(Rct·Cdl)` at max(−Z'') → `Cdl = 1/(2*pi*f_apex*Rct)`.
  This is an estimate (apex from a discrete grid); the circuit fit is exact.
- **Warburg flag**: 45-degree low-frequency tail (median d(−Z'')/dZ' near 1).
- **Exchange current density** (when electrode area is given):
  `j0 = R*T / (n*F*Rct*A)`. Without area, j0 is None + a note.

## 3. Tier 2 — equivalent-circuit fit (optional, impedance.py)

Randles circuit `R0-p(R1,CPE1)` (or `…-W1` when Warburg is detected), fit by
complex nonlinear least squares. The initial guess is seeded from the Tier-1
readout (Rs, Rct, Cdl), which makes divergence rare. CPE (constant-phase
element) is used instead of an ideal capacitor to absorb electrode-surface
non-ideality (exponent n ≤ 1). A normalised chi-square is reported. A diverging
or unavailable fit returns an error field instead of crashing.

## 4. Edge cases

- impedance.py missing → fit returns an error note; Tier 1 still works.
- Warburg tail present → Rct uses the arc-close intercept, not the last point.
- electrode area unknown → j0 None + note.
- Z'' supplied positive → sign normalised.

## References

- Randles, J.E.B., Discuss. Faraday Soc. 1 (1947) 11, DOI 10.1039/df9470100011.
- Lasia, A., "Electrochemical Impedance Spectroscopy and its Applications".
- Gamry / Metrohm / Pine application notes on Randles circuits and Warburg.
- impedance.py — Murbach et al., J. Open Source Softw. 5 (2020) 2349.
