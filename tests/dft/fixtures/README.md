# QE output fixtures — dft.qe_parser golden test

Drop nAM's 19 ground-truth QE **7.4.x** `.out` files here (the ones `qe_parser.py`
was validated against). `test_qe_parser_golden.py` activates automatically when any
`*.out` is present, and skips otherwise (never fails CI for missing data).

Golden values to pin (from the DFT master spec):
- `parse_final_structure`    → 12 atoms, volume ≈ 178.6 Å³
- `parse_scf_summary` (PBE)  → gap ≈ 2.72 eV, JOB DONE, ~49 SCF iters
- `parse_bands`              → 422 k-points, 100 bands
- `band_gap_from_eigenvalues`→ PBE0 gap ≈ 1.16 eV

DOS/PDOS data lives in separate `.dos` / `.pdos_*` files (not `.out`) — parsing
those is DFT **P2** (`output_parser`), and needs those files too.
