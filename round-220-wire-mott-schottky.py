#!/usr/bin/env python3
"""round-220-wire-mott-schottky.py

Wire the new pec_mott_schottky parser into src/main.py dispatch.
Worker convention: patch script allowed (vs app = manual copy).

Two edits:
  1. Add MS-specific metadata extraction next to the pec_jv block.
  2. Add an `elif spectrum_type == "pec_mott_schottky"` dispatch case.

Idempotent: re-running detects existing markers and skips.

Run:
  cp /mnt/d/labbook-patches/round-220-wire-mott-schottky.py ~/LAB-MANAGER/labyra-spectra-worker/
  cd ~/LAB-MANAGER/labyra-spectra-worker && python round-220-wire-mott-schottky.py
"""

from __future__ import annotations

import sys
from pathlib import Path

MAIN = Path("src/main.py")

# --- anchors (verified byte-for-byte against origin/main HEAD) ---
META_ANCHOR = (
    '    pec_bias = metadata.get("appliedBias") or metadata.get("applied_bias_v")\n'
)
META_ADD = (
    '    pec_bias = metadata.get("appliedBias") or metadata.get("applied_bias_v")\n'
    '    pec_eps_r = metadata.get("dielectricConstant") or metadata.get("eps_r")\n'
    '    pec_ms_temp = float(\n'
    '        metadata.get("temperatureK") or metadata.get("temperature_k") or 298.15\n'
    '    )\n'
    '    pec_ms_freqs = metadata.get("frequenciesHz") or metadata.get("frequencies_hz")\n'
)

DISPATCH_ANCHOR = """    elif spectrum_type == "pec_jv":
        from src.parsers.pec_jv import parse_pec_jv
        _a = float(pec_area) if pec_area else None
        _lp = float(pec_light_power) if pec_light_power else None
        _bias = float(pec_bias) if pec_bias else None
        parser = lambda raw: parse_pec_jv(
            raw, area_cm2=_a, light_power_mw_cm2=_lp, applied_bias_v=_bias,
        )
"""

DISPATCH_ADD = DISPATCH_ANCHOR + """    elif spectrum_type == "pec_mott_schottky":
        from src.parsers.pec_mott_schottky import parse_pec_mott_schottky
        _a = float(pec_area) if pec_area else None
        _eps = float(pec_eps_r) if pec_eps_r else None
        _ph = float(lsv_ph) if lsv_ph is not None else None
        parser = lambda raw: parse_pec_mott_schottky(
            raw, eps_r=_eps, reference=lsv_ref, ph=_ph, area_cm2=_a,
            temperature_k=pec_ms_temp, frequencies_hz=pec_ms_freqs,
        )
"""


def main() -> int:
    if not MAIN.exists():
        print(f"ERROR: {MAIN} not found. Run from worker repo root.", file=sys.stderr)
        return 1
    src = MAIN.read_text()

    if "pec_mott_schottky" in src:
        print("Already wired (found 'pec_mott_schottky'). Nothing to do.")
        return 0

    # Edit 1: metadata extraction
    if META_ANCHOR not in src:
        print("ERROR: metadata anchor (pec_bias line) not found. Aborting.", file=sys.stderr)
        return 2
    if src.count(META_ANCHOR) != 1:
        print(f"ERROR: pec_bias anchor appears {src.count(META_ANCHOR)}x (need 1).", file=sys.stderr)
        return 2
    src = src.replace(META_ANCHOR, META_ADD, 1)

    # Edit 2: dispatch case
    if DISPATCH_ANCHOR not in src:
        print("ERROR: pec_jv dispatch anchor not found. Aborting.", file=sys.stderr)
        return 3
    src = src.replace(DISPATCH_ANCHOR, DISPATCH_ADD, 1)

    MAIN.write_text(src)
    print("OK: wired pec_mott_schottky into src/main.py")
    print("  + metadata: pec_eps_r, pec_ms_temp, pec_ms_freqs")
    print("  + dispatch: elif spectrum_type == 'pec_mott_schottky'")
    print("\nNext:")
    print("  ruff check src/main.py src/parsers/pec_mott_schottky.py")
    print("  python -c \"from src.parsers.pec_mott_schottky import parse_pec_mott_schottky; print('import OK')\"")
    print("  bash deploy.sh   # -> revision 00094")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
