# Band Structure & DOS — Results Extraction

Turns raw Quantum ESPRESSO output (`*.out`, `*.dos`) into structured scientific
results: relaxed structure, total energy, Fermi level, **band gap (VBM/CBM,
direct vs indirect)**, and density of states at the Fermi level. This is the
"usable results" layer of the DFT pipeline (input → compute → **results**).

Implementation: `src/dft/qe_parser.py` (`summarize_results`, `band_gap_from_eigenvalues`,
`parse_dos`, `parse_final_structure`, `parse_scf_summary`). Wired in
`src/dft/driver.py` (`_summarize_completed`) → stored in the workflow doc
(`results`) when `overallStatus == "completed"`.

## 1. Band gap from eigenvalues

For a non-spin-polarised insulator/semiconductor with `N` valence electrons,
the number of occupied bands is `n_occ = N / 2` (each band holds 2 electrons).
Over all sampled k-points:

- **VBM** (valence band maximum) = max over k of band `n_occ` (1-indexed) =
  `max_k ε(n_occ, k)`
- **CBM** (conduction band minimum) = min over k of band `n_occ + 1` =
  `min_k ε(n_occ+1, k)`
- **Gap** `E_g = E_CBM − E_VBM`

**Direct vs indirect:** the gap is *direct* if VBM and CBM occur at the **same**
k-point (`|k_VBM − k_CBM| < 1e-4`), otherwise *indirect*. The k-positions of VBM
and CBM are reported so the user can see *where* in the Brillouin zone the band
edges sit.

`N` (`number of electrons`) is read from the scf/nscf output; eigenvalues are read
from the `bands` calculation along the high-symmetry k-path. The scf/nscf grid also
yields a quick `highest occupied / lowest unoccupied` gap (`scfGap`) — useful as a
cross-check, but the k-path `bands` result is the one that resolves direct/indirect.

### Physics notes / edge cases
- **Semilocal underestimation:** PBE / PBE+U systematically underestimates gaps
  (self-interaction, missing derivative discontinuity). E.g. bulk 2H-WS₂ comes out
  ~0.97 eV (indirect Γ→T) vs experimental ~1.3 eV. For accurate gaps use HSE06 or GW.
- **Metals:** `E_VBM > E_CBM` (negative gap) or partially filled bands → the
  "gap" is not physical; treat `dosAtFermi > 0` as the metallic signature.
- **Indirect character is layer-dependent:** bulk TMDCs (2H-MoS₂/WS₂) are indirect;
  monolayers become direct (K–K). VBM at Γ in bulk is the interlayer-coupling fingerprint.
- **Spin-polarised** (`nspin=2`): pass `spin_polarized=True` so `n_occ = N` (bands
  are per-spin); not yet exercised in the bulk workflows.

## 2. Density of states (`parse_dos`)

`dos.x` writes `fildos` with columns `E (eV)`, `dos(E)`, `Int dos(E)` and the Fermi
energy in the header (`EFermi = … eV`). `parse_dos` returns the arrays plus
`dos_at_fermi` (DOS at the energy grid-point nearest E_F) — the metallic/insulating
discriminator and the basis for a DOS plot.

## 3. Relaxed structure (`parse_final_structure`)

From the `Begin final coordinates … End final coordinates` block of a `vc-relax`:
lattice vectors (`CELL_PARAMETERS (alat= …)`, converted bohr→Å with
`BOHR_TO_ANG = 0.529177210903`), cell volume, and fractional positions. `summarize_results`
derives `a = |a₁|`, `c = |a₃|`, `c/a`, and volume. For layered materials the **c/a
contraction** after a vdW-D3 relaxation (e.g. 2H-WS₂ 4.45 → 3.93, ~12 %) is the
dispersion-binding signature.

## References
- Perdew, Burke, Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996) — PBE. DOI:10.1103/PhysRevLett.77.3865
- Heyd, Scuseria, Ernzerhof, *J. Chem. Phys.* **118**, 8207 (2003) — HSE. DOI:10.1063/1.1564060
- Grimme et al., *J. Chem. Phys.* **132**, 154104 (2010) — DFT-D3. DOI:10.1063/1.3382344
- Kuc, Zibouche, Heine, *Phys. Rev. B* **83**, 245213 (2011) — TMDC indirect→direct. DOI:10.1103/PhysRevB.83.245213
