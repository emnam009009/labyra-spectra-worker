# QE fixtures — dft.qe_parser golden test

Two layers (see `tests/dft/test_qe_parser_golden.py`):

## 1. Synthetic (committed, always run)
`synthetic_scf.out` · `synthetic_relax.out` · `synthetic_bands.out` — tiny,
format-correct QE 7.4.x snippets that exercise every parser. No large files
needed; these run in CI as-is.

## 2. Real (optional — your files, NOT committed)
Real QE `.out` files are **huge**, so they are **gitignored** here (`.gitignore`
keeps only `synthetic_*.out`). Drop your actual files in this folder and the
golden test picks them up automatically with **invariant checks** (gap ≥ 0,
energy < 0, consistent band count, volume > 0). You don't need to send them to
anyone — they only run locally / in your CI.

Pin **exact values** per file with an optional `expected.json`:
```json
{
  "my_scf.out":   { "band_gap_ev": 2.72, "scf_iterations": 49 },
  "my_relax.out": { "n_atoms": 12 },
  "my_bands.out": { "nks": 422 }
}
```

### Too big to keep even locally?
`python3 scripts/trim_qe_out.py big.out --kblocks 6 -o tests/dft/fixtures/my_bands.out`
trims a megabyte .out to a few KB (header markers + summary/convergence + final
coordinates + first N band k-blocks) that still parses. The trimmed file's
declared k-count may exceed the parsed count — that's fine (invariants use the
parsed count; only set `"nks"` in expected.json for an untrimmed file).
