#!/usr/bin/env python3
"""
trim_qe_out.py — shrink a (huge) QE pw.x .out into a tiny golden fixture that
still parses with src/dft/qe_parser.py. Keeps the header markers, summary +
convergence lines, the Begin/End final-coordinates block, and the first N bands
k-blocks (whole). Lets you commit a real-derived fixture without the megabytes.

Usage:
  python3 scripts/trim_qe_out.py big.out                      > tests/dft/fixtures/my_scf.out
  python3 scripts/trim_qe_out.py big_bands.out --kblocks 6 -o tests/dft/fixtures/my_bands.out

@phase R272w-d (DFT P0)
"""
from __future__ import annotations

import argparse
import re
import sys

_KEEP = re.compile(
    r"(PWSCF v\.|lattice parameter \(alat\)|number of electrons|number of Kohn-Sham states|"
    r"number of k points|estimated scf accuracy|^!\s+total energy|Total force\s+=|"
    r"highest occupied|the Fermi energy is|convergence has been achieved|"
    r"bfgs converged|JOB DONE)",
    re.M,
)
_KHEAD = re.compile(r"k =\s*-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\s*\(\s*\d+ PWs\)\s+bands \(ev\)")


def trim(text: str, kblocks: int = 6) -> str:
    lines = text.split("\n")
    kidx = [i for i, ln in enumerate(lines) if _KHEAD.search(ln)]
    head_end = kidx[0] if kidx else len(lines)

    kept: list[str] = [ln for ln in lines[:head_end] if _KEEP.search(ln)]

    if kidx:  # keep the first `kblocks` whole k-blocks
        end = kidx[kblocks] if len(kidx) > kblocks else len(lines)
        kept += [""] + lines[kidx[0]:end]

    fc = re.search(r"Begin final coordinates.*?End final coordinates", text, re.S)
    if fc:
        kept += ["", fc.group(0)]

    body = "\n".join(kept)
    if "JOB DONE." not in body:
        body += "\n\n     JOB DONE."
    return body + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("--kblocks", type=int, default=6, help="bands k-blocks to keep (default 6)")
    ap.add_argument("-o", "--out", help="output path (default: stdout)")
    args = ap.parse_args()
    with open(args.infile, encoding="utf-8", errors="ignore") as f:
        trimmed = trim(f.read(), args.kblocks)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(trimmed)
        sys.stderr.write(f"wrote {args.out} ({len(trimmed)} chars)\n")
    else:
        sys.stdout.write(trimmed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
