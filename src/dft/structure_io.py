"""
structure_io.py — CIF/POSCAR/bất kỳ → QE input. Thay VESTA→POSCAR thủ công.

TRIẾT LÝ TIN CẬY TUYỆT ĐỐI: không TRUST map ibrav, mà VERIFY.
  1. Mặc định ibrav=0 + CELL_PARAMETERS (robust — QE tự dò đối xứng).
  2. ibrav≠0 CHỈ thử cho hệ P đối xứng cao (whitelist hẹp), và CHỈ dùng nếu
     reconstruct lại cell từ (ibrav, celldm) KHỚP cell gốc (round-trip verify).
  3. Verify thất bại (vd Si fcc centered) → tự fallback ibrav=0, KHÔNG sai ngầm.
  4. Sanity checks (volume, composition, atom chồng) → vô lý thì RAISE, không nuốt.
"""
import math

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

BOHR = 0.529177210903

# Whitelist: CHỈ hệ P đối xứng cao (primitive = conventional, celldm thẳng)
SAFE_IBRAV = {("cubic", "P"): 1, ("hexagonal", "P"): 4,
              ("tetragonal", "P"): 6, ("orthorhombic", "P"): 8}


def _cell_from_ibrav(ibrav, cd):
    """Dựng lại 3x3 cell (Å) từ ibrav + celldm — chỉ cho hệ P whitelist."""
    a = cd[1] * BOHR
    if ibrav == 1:
        return np.array([[a, 0, 0], [0, a, 0], [0, 0, a]])
    if ibrav == 4:
        c = cd[3] * a
        return np.array([[a, 0, 0], [-a / 2, a * math.sqrt(3) / 2, 0], [0, 0, c]])
    if ibrav == 6:
        c = cd[3] * a
        return np.array([[a, 0, 0], [0, a, 0], [0, 0, c]])
    if ibrav == 8:
        b, c = cd[2] * a, cd[3] * a
        return np.array([[a, 0, 0], [0, b, 0], [0, 0, c]])
    return None


def _cells_match(recon, latt, rtol=1e-4, atol_deg=0.05):
    """So abc + góc giữa cell reconstruct và cell gốc."""
    r = Lattice(recon)
    abc_ok = np.allclose(r.abc, latt.abc, rtol=rtol)
    ang_ok = np.allclose(r.angles, latt.angles, atol=atol_deg)
    return bool(abc_ok and ang_ok)


def _sanity(s):
    """Kiểm tra vô lý → trả danh sách lỗi (rỗng = sạch)."""
    errs = []
    if s.volume <= 0:
        errs.append(f"volume <= 0 ({s.volume})")
    # khoảng cách min giữa nguyên tử
    try:
        dm = s.distance_matrix
        np.fill_diagonal(dm, np.inf)
        if dm.min() < 0.5:
            errs.append(f"hai nguyên tử quá gần ({dm.min():.3f} Å < 0.5) — có thể chồng")
    except Exception as e:
        errs.append(f"không tính được distance matrix: {e}")
    # occupancy = 1 (không disorder)
    for site in s:
        if abs(site.species.num_atoms - 1.0) > 1e-6:
            errs.append("site có occupancy ≠ 1 (disorder) — cần xử lý thủ công")
            break
    return errs


def structure_to_qe(struct, use_primitive=True, prefer_ibrav=True,
                    angle_tol=5, symprec=1e-3):
    sga = SpacegroupAnalyzer(struct, symprec=symprec, angle_tolerance=angle_tol)
    s = (sga.get_primitive_standard_structure() if use_primitive
         else sga.get_conventional_standard_structure())
    sga2 = SpacegroupAnalyzer(s, symprec=symprec, angle_tolerance=angle_tol)
    cs = sga2.get_crystal_system()
    sg = f"{sga2.get_space_group_symbol()} (#{sga2.get_space_group_number()})"
    centering = sga2.get_space_group_symbol()[0]

    ibrav, celldm, verified, note = 0, None, None, "ibrav=0 (mặc định robust)"
    key = (cs, centering)
    if prefer_ibrav and key in SAFE_IBRAV:
        cand = SAFE_IBRAV[key]
        a, b, c = s.lattice.abc
        cd = {1: a / BOHR}
        if cand in (6,): cd[3] = c / a
        if cand == 4: cd[3] = c / a
        if cand == 8: cd[2] = b / a; cd[3] = c / a
        recon = _cell_from_ibrav(cand, cd)
        if recon is not None and _cells_match(recon, s.lattice):
            ibrav, celldm, verified, note = cand, cd, True, f"ibrav={cand} (đã verify round-trip)"
        else:
            verified, note = False, f"ibrav={cand} reconstruct LỆCH → fallback ibrav=0"

    sanity = _sanity(s)
    return {
        "crystal_system": cs, "space_group": sg, "centering": centering,
        "n_atoms": len(s), "ibrav": ibrav, "celldm": celldm,
        "cell_ang": s.lattice.matrix, "structure": s,
        "verified": verified, "note": note, "sanity_errors": sanity,
    }


def emit_qe(res, decimals=9):
    """Sinh &SYSTEM-lattice + ATOMIC_POSITIONS (+CELL_PARAMETERS nếu ibrav=0, đủ chữ số)."""
    if res["sanity_errors"]:
        raise ValueError("SANITY FAIL (không sinh input): " + "; ".join(res["sanity_errors"]))
    s = res["structure"]
    sysl = []
    if res["ibrav"]:
        sysl.append(f"    ibrav = {res['ibrav']}")
        for k in sorted(res["celldm"]):
            sysl.append(f"    celldm({k}) = {res['celldm'][k]:.6f}")
    else:
        sysl.append("    ibrav = 0")
    sysl += [f"    nat = {len(s)}",
             f"    ntyp = {len(set(a.symbol for a in s.species))}"]
    body = []
    if res["ibrav"] == 0:
        body.append("CELL_PARAMETERS angstrom")
        for row in res["cell_ang"]:
            body.append(("  {:." + str(decimals) + "f} {:." + str(decimals) +
                         "f} {:." + str(decimals) + "f}").format(*row))
    body.append("ATOMIC_POSITIONS crystal")
    for site in s:
        body.append(("{:3s} {:." + str(decimals) + "f} {:." + str(decimals) +
                     "f} {:." + str(decimals) + "f}").format(site.species_string, *site.frac_coords))
    return "\n".join(sysl), "\n".join(body)


def cif_to_qe(path, **kw):
    return structure_to_qe(Structure.from_file(path), **kw)
