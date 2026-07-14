"""
qe_parser.py — Parser cho Quantum ESPRESSO 7.4.x output (.out từ pw.x)
Viết riêng cho QE 7.4.x, test trên file thật h-WO3 bulk.
Phạm vi: structure handoff (relax→scf), convergence (monitoring), scf summary, bands.
KHÔNG dùng cho dos.x/projwfc.x output (format khác — parser riêng).
"""
import math
import re

BOHR_TO_ANG = 0.529177210903
AMU_TO_G = 1.66053906660e-24

# Standard atomic weights (g/mol ≈ amu) — enough for common inorganic materials.
ATOMIC_MASS = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180, "Na": 22.990, "Mg": 24.305,
    "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45, "Ar": 39.948,
    "K": 39.098, "Ca": 40.078, "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996,
    "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904, "Kr": 83.798,
    "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224, "Nb": 92.906, "Mo": 95.95,
    "Ru": 101.07, "Rh": 102.91, "Pd": 106.42, "Ag": 107.87, "Cd": 112.41, "In": 114.82,
    "Sn": 118.71, "Sb": 121.76, "Te": 127.60, "I": 126.90, "Xe": 131.29, "Cs": 132.91,
    "Ba": 137.33, "La": 138.91, "Ce": 140.12, "Hf": 178.49, "Ta": 180.95, "W": 183.84,
    "Re": 186.21, "Os": 190.23, "Ir": 192.22, "Pt": 195.08, "Au": 196.97, "Hg": 200.59,
    "Tl": 204.38, "Pb": 207.2, "Bi": 208.98,
}


def _density_g_cm3(species, volume_ang3):
    """Crystal density (g/cm³) from the relaxed cell: Σ atomic mass / unit-cell
    volume. Returns None if the volume is missing or any element is unknown."""
    if not species or not volume_ang3 or volume_ang3 <= 0:
        return None
    try:
        total_amu = sum(ATOMIC_MASS[s] for s in species)
    except KeyError:
        return None  # unknown element → don't guess
    # g/cm³ = (Σ amu · g/amu) / (V · cm³/Å³);  1e-24 cm³/Å³ cancels AMU_TO_G's 1e-24
    return round(total_amu * AMU_TO_G / (volume_ang3 * 1e-24), 3)


def _walltime_to_seconds(s):
    """Parse a QE wall-time token ('48.90s', '2m20.12s', '1h27m', '3h20m15s')."""
    m = re.fullmatch(r"(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?\s*(?:([\d.]+)\s*s)?", s.strip())
    if not m or not any(m.groups()):
        return None
    h = int(m.group(1)) if m.group(1) else 0
    mins = int(m.group(2)) if m.group(2) else 0
    sec = float(m.group(3)) if m.group(3) else 0.0
    val = h * 3600 + mins * 60 + sec
    return round(val, 2) if val > 0 else None


def parse_walltime(text):
    """Total wall-clock seconds from the QE timing footer — the program-total
    line ('PWSCF : … CPU … WALL', also DOS/PROJWFC/BANDS), which unlike the
    per-routine lines has no '( N calls)' suffix. Returns None if absent."""
    total = None
    for m in re.finditer(
        r"^\s*([A-Z][A-Z0-9_]+)\s*:\s+.*?CPU\s+(\S.*?)\s+WALL\s*$", text, re.M
    ):
        secs = _walltime_to_seconds(m.group(2))
        if secs is not None:
            total = secs  # keep the last (final program total)
    return total


def _floats(line):
    """True nếu dòng chỉ gồm các số (dùng để gom eigenvalues)."""
    toks = line.split()
    if not toks:
        return None
    try:
        return [float(t) for t in toks]
    except ValueError:
        return None


def parse_scf_summary(text):
    """Tóm tắt 1 run scf/nscf: năng lượng, HOMO/LUMO/gap, Fermi, hội tụ, JOB DONE."""
    out = {
        "total_energy_ry": None, "homo_ev": None, "lumo_ev": None,
        "band_gap_ev": None, "fermi_ev": None, "scf_iterations": None,
        "n_electrons": None, "nbnd": None, "alat_bohr": None, "job_done": False,
        "total_mag": None, "abs_mag": None, "spin_polarized": False,
    }
    # total energy: lấy giá trị CUỐI (sau cùng) — run cuối là trên cấu trúc tối ưu
    te = re.findall(r"^!\s+total energy\s+=\s+(-?[\d.]+)\s+Ry", text, re.M)
    if te:
        out["total_energy_ry"] = float(te[-1])

    m = re.search(r"highest occupied, lowest unoccupied level \(ev\):\s+(-?[\d.]+)\s+(-?[\d.]+)", text)
    if m:
        out["homo_ev"], out["lumo_ev"] = float(m.group(1)), float(m.group(2))
        out["band_gap_ev"] = round(out["lumo_ev"] - out["homo_ev"], 4)
    else:
        m2 = re.search(r"highest occupied level \(ev\):\s+(-?[\d.]+)", text)
        if m2:
            out["homo_ev"] = float(m2.group(1))

    m = re.search(r"the Fermi energy is\s+(-?[\d.]+)\s*ev", text)
    if m:
        out["fermi_ev"] = float(m.group(1))

    m = re.search(r"convergence has been achieved in\s+(\d+) iterations", text)
    if m:
        out["scf_iterations"] = int(m.group(1))

    m = re.search(r"number of electrons\s+=\s+([\d.]+)", text)
    if m:
        out["n_electrons"] = float(m.group(1))
    m = re.search(r"number of Kohn-Sham states\s*=\s+(\d+)", text)
    if m:
        out["nbnd"] = int(m.group(1))
    m = re.search(r"lattice parameter \(alat\)\s+=\s+([\d.]+)", text)
    if m:
        out["alat_bohr"] = float(m.group(1))

    mm = re.findall(r"total magnetization\s+=\s+(-?[\d.]+)\s+Bohr mag/cell", text)
    if mm:
        out["total_mag"] = float(mm[-1])
    ma = re.findall(r"absolute magnetization\s+=\s+([\d.]+)\s+Bohr mag/cell", text)
    if ma:
        out["abs_mag"] = float(ma[-1])
    out["spin_polarized"] = bool(mm)

    out["job_done"] = "JOB DONE." in text
    return out


def parse_convergence(text):
    """Chuỗi hội tụ cho monitoring: scf accuracy + (energy, force) mỗi ionic step + đã hội tụ?
    Live-stream-aware: chịu được .out đang chạy dở (partial); `scf_seconds` là mốc
    'total cpu time spent up to now' của TỪNG iteration (trục thời gian), `job_done`
    phân biệt file cuối với snapshot giữa chừng."""
    scf_acc = [float(x) for x in re.findall(
        r"estimated scf accuracy\s+<\s+([0-9.eE+-]+)\s+Ry", text)]
    scf_seconds = [float(x) for x in re.findall(
        r"total cpu time spent up to now is\s+([\d.]+)\s+secs", text)]
    energies = [float(x) for x in re.findall(
        r"^!\s+total energy\s+=\s+(-?[\d.]+)\s+Ry", text, re.M)]
    forces = [float(x) for x in re.findall(
        r"Total force\s+=\s+([\d.]+)\s+Total SCF correction", text)]
    ionic = [{"energy_ry": e, "total_force": f} for e, f in zip(energies, forces, strict=True)]

    bfgs = re.search(r"bfgs converged in\s+(\d+) scf cycles and\s+(\d+) bfgs steps", text)
    converged = bool(bfgs) or ("convergence has been achieved" in text)
    return {
        "scf_accuracy": scf_acc,
        "scf_seconds": scf_seconds,
        "ionic_steps": ionic,
        "n_ionic_steps": len(ionic),
        "converged": converged,
        "job_done": "JOB DONE" in text,
        "bfgs_steps": int(bfgs.group(2)) if bfgs else None,
        "final_force": forces[-1] if forces else None,
        "final_scf_accuracy": scf_acc[-1] if scf_acc else None,
    }


def parse_final_structure(text):
    """Structure handoff: trích cấu trúc TỐI ƯU từ block 'Begin/End final coordinates'.
    Trả None nếu không có (file không phải relax/vc-relax)."""
    m = re.search(r"Begin final coordinates(.*?)End final coordinates", text, re.S)
    if not m:
        return None
    block = m.group(1)

    vol = re.search(r"new unit-cell volume\s+=\s+[\d.]+ a\.u\.\^3 \(\s+([\d.]+) Ang", block)
    volume_ang = float(vol.group(1)) if vol else None

    cm = re.search(
        r"CELL_PARAMETERS \(alat=\s*([\d.]+)\)\s*\n"
        r"((?:\s*-?[\d.]+\s+-?[\d.]+\s+-?[\d.]+\s*\n){3})",
        block,
    )
    cell_ang = None
    alat_bohr = None
    if cm:
        alat_bohr = float(cm.group(1))
        rows = [[float(x) for x in r.split()] for r in cm.group(2).strip().split("\n")]
        cell_ang = [[v * alat_bohr * BOHR_TO_ANG for v in row] for row in rows]

    species, frac = [], []
    am = re.search(r"ATOMIC_POSITIONS \(crystal\)\s*\n(.*?)(?:\nEnd final|$)", block, re.S)
    if am:
        for line in am.group(1).strip().split("\n"):
            toks = line.split()
            if len(toks) >= 4:
                species.append(toks[0])
                frac.append([float(toks[1]), float(toks[2]), float(toks[3])])

    return {
        "alat_bohr": alat_bohr,
        "cell_ang": cell_ang,
        "species": species,
        "frac_positions": frac,
        "n_atoms": len(species),
        "volume_ang3": volume_ang,
    }


def parse_bands(text):
    """Band eigenvalues: list k-point + eigenvalues (eV) mỗi k. Cho band-structure plot."""
    nks = None
    m = re.search(r"number of k points\s*=\s+(\d+)", text)
    if m:
        nks = int(m.group(1))

    lines = text.split("\n")
    khead = re.compile(r"k =\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*\(\s*\d+ PWs\)\s+bands \(ev\)")
    kpoints, eigenvalues = [], []
    i = 0
    while i < len(lines):
        m = khead.search(lines[i])
        if m:
            kpoints.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
            j = i + 1
            while j < len(lines) and not lines[j].strip():  # bỏ blank
                j += 1
            ev = []
            while j < len(lines):
                vals = _floats(lines[j])
                if vals is None:  # blank/non-float → hết block
                    break
                ev.extend(vals)
                j += 1
            eigenvalues.append(ev)
            i = j
        else:
            i += 1
    return {
        "nks_declared": nks,
        "nks_parsed": len(kpoints),
        "nbnd": len(eigenvalues[0]) if eigenvalues else None,
        "kpoints": kpoints,
        "eigenvalues": eigenvalues,
    }


def to_qe_structure_block(struct):
    """Sinh block CELL_PARAMETERS + ATOMIC_POSITIONS cho input QE bước sau (handoff)."""
    if not struct or not struct["cell_ang"]:
        return None
    lines = ["CELL_PARAMETERS (angstrom)"]
    for row in struct["cell_ang"]:
        lines.append("  {:14.9f} {:14.9f} {:14.9f}".format(*row))
    lines.append("ATOMIC_POSITIONS (crystal)")
    for sp, p in zip(struct["species"], struct["frac_positions"], strict=True):
        lines.append("{:4s} {:14.9f} {:14.9f} {:14.9f}".format(sp, *p))
    return "\n".join(lines)


def band_gap_from_eigenvalues(bands_result, n_electrons, spin_polarized=False):
    """Tính band gap từ eigenvalues (robust cho hybrid/smearing — khi summary chỉ in Fermi).
    Insulator không spin: n_occupied = n_electrons / 2.
    VBM = max(occupied) qua mọi k; CBM = min(unoccupied) qua mọi k."""
    ev = bands_result["eigenvalues"]
    kpts = bands_result.get("kpoints") or [None] * len(ev)
    if not ev or not n_electrons:
        return None
    n_occ = int(round(n_electrons / (1 if spin_polarized else 2)))
    if any(len(e) <= n_occ for e in ev):
        return None
    vbm_i = max(range(len(ev)), key=lambda i: ev[i][n_occ - 1])
    cbm_i = min(range(len(ev)), key=lambda i: ev[i][n_occ])
    vbm, cbm = ev[vbm_i][n_occ - 1], ev[cbm_i][n_occ]
    vbm_k, cbm_k = kpts[vbm_i], kpts[cbm_i]
    direct = None
    if vbm_k is not None and cbm_k is not None:
        direct = all(abs(a - b) < 1e-4 for a, b in zip(vbm_k, cbm_k, strict=True))
    return {
        "vbm_ev": round(vbm, 4), "cbm_ev": round(cbm, 4),
        "band_gap_ev": round(cbm - vbm, 4),
        "vbm_k": [round(x, 5) for x in vbm_k] if vbm_k else None,
        "cbm_k": [round(x, 5) for x in cbm_k] if cbm_k else None,
        "direct": direct,
    }


def parse_dos(text):
    """Parse QE dos.x fildos output: cột E(eV), dos(E), integrated dos(E).
    Trả energies/dos/integrated + EFermi (nếu có trong header) + dos tại Fermi."""
    fermi = None
    m = re.search(r"EFermi\s*=\s*(-?\d+\.\d+)", text)
    if m:
        fermi = float(m.group(1))
    energies, dos, idos = [], [], []
    for line in text.split("\n"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        vals = _floats(s)
        if vals and len(vals) >= 2:
            energies.append(vals[0])
            dos.append(vals[1])
            if len(vals) >= 3:
                idos.append(vals[2])
    dos_at_fermi = None
    if fermi is not None and energies:
        ix = min(range(len(energies)), key=lambda i: abs(energies[i] - fermi))
        dos_at_fermi = round(dos[ix], 6)
    return {
        "fermi_ev": fermi,
        "energies_ev": energies,
        "dos": dos,
        "integrated_dos": idos or None,
        "dos_at_fermi": dos_at_fermi,
        "n_points": len(energies),
    }


_PDOS_FN_RE = re.compile(r"pdos_atm#(\d+)\(([A-Za-z]+)\)_wfc#(\d+)\(([spdfSPDF])\)")


def parse_pdos(files):
    """Aggregate projwfc PDOS files by (element, orbital l).

    files: dict {filename: text}. Each projwfc file has col0=E(eV), col1=ldos(E)
    (already summed over m), col2+=per-m pdos. col1 is accumulated across every
    file sharing the same (element, l) — over all atoms of that element and all
    of their wfc shells of that l. Returns energies + one series per (element, l),
    sorted by element then s<p<d<f."""
    energies = []
    groups = {}
    order = []
    for fname, text in files.items():
        m = _PDOS_FN_RE.search(fname)
        if not m:
            continue
        elem = m.group(2)
        l = m.group(4).lower()
        label = f"{elem}-{l}"
        es, ld = [], []
        for line in text.split("\n"):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            vals = _floats(s)
            if vals and len(vals) >= 2:
                es.append(vals[0])
                ld.append(vals[1])
        if not ld:
            continue
        if not energies:
            energies = es
        if label not in groups:
            groups[label] = [0.0] * len(ld)
            order.append(label)
        g = groups[label]
        n = min(len(g), len(ld))
        for i in range(n):
            g[i] += ld[i]

    def _key(lab):
        elem, orb = lab.split("-")
        return (elem, "spdf".index(orb) if orb in "spdf" else 9)

    labels = sorted(order, key=_key)
    series = [{"label": lab, "dos": [round(x, 6) for x in groups[lab]]} for lab in labels]
    return {"energies_ev": energies, "pdos": series, "n_points": len(energies)}


def pdos_character(energies, pdos_series, target_ev, top=4):
    """Orbital character at an energy (e.g. VBM/CBM): % each (element,l) contributes.
    Returns [{label, pct}] sorted desc, top entries with pct>0."""
    if not energies or not pdos_series:
        return []
    i = min(range(len(energies)), key=lambda j: abs(energies[j] - target_ev))
    vals = []
    for s in pdos_series:
        d = s.get("dos") or []
        vals.append((s.get("label", "?"), d[i] if i < len(d) else 0.0))
    tot = sum(v for _, v in vals)
    if tot <= 0:
        return []
    contribs = [(lab, v / tot * 100.0) for lab, v in vals if v > 0]
    contribs.sort(key=lambda x: -x[1])
    return [{"label": lab, "pct": round(pct, 1)} for lab, pct in contribs[:top]]


def summarize_results(outputs):
    """Tổng hợp kết quả khoa học có cấu trúc từ text .out các unit (input → usable results).
    outputs: dict {role: out_text} với role ∈ {vc-relax|relax, scf|nscf, bands, dos}.
    Trả: relaxedStructure (a/c/V/density), totalEnergyRy, fermiEv, nElectrons,
         scfGap (HOMO/LUMO trên lưới scf), bandGap (VBM/CBM/k/direct từ bands k-path)."""
    res: dict = {}

    relax_txt = outputs.get("vc-relax") or outputs.get("relax")
    if relax_txt:
        fs = parse_final_structure(relax_txt)
        if fs and fs.get("cell_ang"):
            cell = fs["cell_ang"]
            a = math.sqrt(sum(v * v for v in cell[0]))
            c = math.sqrt(sum(v * v for v in cell[2]))
            res["relaxedStructure"] = {
                "aAng": round(a, 4),
                "cAng": round(c, 4),
                "coa": round(c / a, 4) if a else None,
                "volumeAng3": fs.get("volume_ang3"),
                "nAtoms": fs.get("n_atoms"),
                "density": _density_g_cm3(fs.get("species") or [], fs.get("volume_ang3")),
            }

    sc_txt = outputs.get("scf") or outputs.get("nscf")
    n_elec = None
    if sc_txt:
        summ = parse_scf_summary(sc_txt)
        res["totalEnergyRy"] = summ.get("total_energy_ry")
        res["fermiEv"] = summ.get("fermi_ev")
        n_elec = summ.get("n_electrons")
        if n_elec:
            res["nElectrons"] = n_elec
        if summ.get("band_gap_ev") is not None:
            res["scfGap"] = {
                "gapEv": summ["band_gap_ev"],
                "homoEv": summ.get("homo_ev"),
                "lumoEv": summ.get("lumo_ev"),
            }

    bands_txt = outputs.get("bands")
    if bands_txt and n_elec:
        bg = band_gap_from_eigenvalues(parse_bands(bands_txt), n_elec)
        if bg:
            res["bandGap"] = bg

    dos_txt = outputs.get("dos")
    if dos_txt:
        d = parse_dos(dos_txt)
        if d.get("dos_at_fermi") is not None:
            res["dosAtFermi"] = d["dos_at_fermi"]

    return res
