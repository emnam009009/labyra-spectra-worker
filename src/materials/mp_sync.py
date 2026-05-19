"""
Materials Project API sync.

Pulls structure, electronic properties, and formation energy from MP.

Strategy (R184-hotfix1):
  1. PREFERRED: Look up by exact mp_id (from R183-2 manual seed) — guarantees
     correct experimentally-relevant polymorph.
  2. FALLBACK: Search by formula — picks lowest energy_above_hull entry.
     Warning: this can return theoretical/metastable polymorphs.

Manual seed always wins for crystal structure (mpId is the source of truth
for which polymorph to fetch).

@phase R184-hotfix1-mp-id-lookup
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Curated mp_id map — experimentally-relevant polymorphs per formula.
# Used when seed doc has no mpId. Format: { formula: (mp_id, phase_note) }
CURATED_MP_IDS: dict[str, tuple[str, str]] = {
    # R183-2 seeded
    "MoS2":  ("mp-2815",   "2H, hexagonal P6_3/mmc"),
    "WS2":   ("mp-224",    "2H, hexagonal P6_3/mmc"),
    "WO3":   ("mp-18905",  "monoclinic P2_1/n at RT"),
    "TiO2":  ("mp-390",    "anatase, tetragonal I4_1/amd"),
    "ZnO":   ("mp-2133",   "wurtzite, hexagonal P6_3mc"),
    # R184 expansion
    "Fe2O3": ("mp-24972",  "hematite, alpha-Fe2O3"),
    "SnO2":  ("mp-856",    "cassiterite, tetragonal"),
    "MoSe2": ("mp-1634",   "2H-MoSe2"),
    "WSe2":  ("mp-1821",   "2H-WSe2"),
    "MoO3":  ("mp-18856",  "alpha-MoO3, orthorhombic"),
    "In2O3": ("mp-22323",  "bixbyite cubic"),
    "Ga2O3": ("mp-886",    "beta-Ga2O3, monoclinic"),
    "BiVO4": ("mp-23878",  "scheelite monoclinic"),
    "CuO":   ("mp-704645", "tenorite, monoclinic"),
    "NiO":   ("mp-19009",  "bunsenite, cubic rocksalt"),
    "CoO":   ("mp-19128",  "rocksalt cubic"),
    "MnO2":  ("mp-510408", "pyrolusite beta-MnO2"),
    "V2O5":  ("mp-25279",  "orthorhombic"),
    "Nb2O5": ("mp-3239",   "H-Nb2O5 monoclinic"),
    "Ta2O5": ("mp-1241894", "orthorhombic"),
}


_SUMMARY_FIELDS = [
    "material_id",
    "formula_pretty",
    "symmetry",
    "energy_above_hull",
    "formation_energy_per_atom",
    "band_gap",
    "is_gap_direct",
    "volume",
    "density",
    "nsites",
    "elements",
    "theoretical",
]


def _conductivity_type(band_gap: float | None) -> str:
    if band_gap is None:
        return "unknown"
    if band_gap == 0.0:
        return "metal"
    if band_gap < 0.5:
        return "semimetal"
    if band_gap < 4.0:
        return "semiconductor"
    return "insulator"


def _doc_to_profile(best: Any) -> dict[str, Any]:
    """Convert MP summary doc → materialProfiles patch dict."""
    mp_id: str = str(getattr(best, "material_id", ""))
    band_gap: float | None = getattr(best, "band_gap", None)
    is_direct: bool | None = getattr(best, "is_gap_direct", None)
    symmetry = getattr(best, "symmetry", None)

    crystal_system = ""
    space_group = ""
    space_group_number = None
    if symmetry:
        crystal_system = str(getattr(symmetry, "crystal_system", "")).lower()
        space_group = str(getattr(symmetry, "symbol", ""))
        space_group_number = getattr(symmetry, "number", None)

    formation_energy = getattr(best, "formation_energy_per_atom", None)
    energy_above_hull = getattr(best, "energy_above_hull", None)
    density = getattr(best, "density", None)
    volume = getattr(best, "volume", None)
    theoretical = getattr(best, "theoretical", None)

    result: dict[str, Any] = {
        "mpId": mp_id,
        "mpData": {
            "energyAboveHull": round(energy_above_hull, 4) if energy_above_hull is not None else None,
            "formationEnergyPerAtom": round(formation_energy, 4) if formation_energy is not None else None,
            "volume": round(volume, 3) if volume is not None else None,
            "density": round(density, 4) if density is not None else None,
            "theoretical": theoretical,
            "syncedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
    }

    # IMPORTANT: only update crystal structure fields when MP id was looked up
    # via exact match (handled by caller); formula-search fallback should NOT
    # overwrite manual seed structure.

    if crystal_system:
        result["_mpCrystalSystem"] = crystal_system
    if space_group:
        result["_mpSpaceGroup"] = space_group
    if space_group_number:
        result["_mpSpaceGroupNumber"] = space_group_number

    if band_gap is not None:
        result["electronicPropsMP"] = {
            "bandgapEv": round(band_gap, 3),
            "bandgapType": "direct" if is_direct else "indirect",
            "bandgapNotes": (
                f"DFT-GGA from Materials Project {mp_id}. "
                "GGA typically underestimates by 30-50%. "
                "Experimental values in electronicProps field."
            ),
            "conductivityType": _conductivity_type(band_gap),
            "citation": {
                "doi": "10.1063/5.0013288",
                "journal": "APL Materials",
                "year": 2020,
                "title": "The Materials Project: A materials genome approach",
                "verified": True,
            },
        }

    return result


def fetch_by_id(mp_id: str, api_key: str) -> dict[str, Any] | None:
    """Fetch a single MP entry by exact material_id."""
    try:
        from mp_api.client import MPRester  # type: ignore[import]
    except ImportError:
        logger.error("mp_api not installed")
        return None

    try:
        with MPRester(api_key) as mpr:
            docs = mpr.summary.search(
                material_ids=[mp_id],
                fields=_SUMMARY_FIELDS,
            )
    except Exception:
        logger.exception("MP fetch_by_id failed: %s", mp_id)
        return None

    if not docs:
        logger.warning("MP: id not found: %s", mp_id)
        return None

    return _doc_to_profile(docs[0])


def fetch_by_formula(formula: str, api_key: str) -> dict[str, Any] | None:
    """
    FALLBACK: search by formula, return lowest energy_above_hull.
    WARNING: may return theoretical/metastable polymorph.
    """
    try:
        from mp_api.client import MPRester  # type: ignore[import]
    except ImportError:
        logger.error("mp_api not installed")
        return None

    try:
        with MPRester(api_key) as mpr:
            docs = mpr.summary.search(
                formula=formula,
                fields=_SUMMARY_FIELDS,
            )
    except Exception:
        logger.exception("MP fetch_by_formula failed: %s", formula)
        return None

    if not docs:
        return None

    def sort_key(d: Any) -> tuple[float, int]:
        hull = getattr(d, "energy_above_hull", 9999) or 9999
        mid = str(getattr(d, "material_id", "mp-999999")).replace("mp-", "")
        return (hull, int(mid) if mid.isdigit() else 999999)

    docs_sorted = sorted(docs, key=sort_key)
    return _doc_to_profile(docs_sorted[0])


def sync_to_firestore(
    formula: str,
    api_key: str,
    db: Any,
) -> dict[str, Any]:
    """
    Sync MP data for formula into /materialProfiles/{formula}.

    Resolution order:
      1. Existing seed doc has mpId → fetch by that exact id
      2. Curated map has entry → fetch by that mp_id
      3. Fall back to formula search (flagged as ambiguous)
    """
    doc_ref = db.collection("materialProfiles").document(formula)
    existing = doc_ref.get()
    existing_data = existing.to_dict() if existing.exists else {}
    seeded_mp_id = existing_data.get("mpId") if existing_data else None

    resolution = "unknown"
    mp_data: dict[str, Any] | None = None

    if seeded_mp_id and seeded_mp_id.startswith("mp-"):
        mp_data = fetch_by_id(seeded_mp_id, api_key)
        resolution = "seeded_id"
    elif formula in CURATED_MP_IDS:
        curated_id, phase_note = CURATED_MP_IDS[formula]
        mp_data = fetch_by_id(curated_id, api_key)
        resolution = f"curated_id ({phase_note})"
    else:
        mp_data = fetch_by_formula(formula, api_key)
        resolution = "formula_search_AMBIGUOUS"
        logger.warning(
            "Using formula search for %s — polymorph may be wrong. "
            "Add to CURATED_MP_IDS for guaranteed correctness.",
            formula,
        )

    if not mp_data:
        return {"formula": formula, "status": "not_found", "resolution": resolution}

    # Strip internal _mp* fields — these are crystal structure from MP that
    # we DO NOT want to overwrite manual seed with. Only merge mpData and
    # electronicPropsMP (additive, namespaced fields).
    safe_patch = {
        "mpId": mp_data["mpId"],
        "mpData": mp_data["mpData"],
    }
    if "electronicPropsMP" in mp_data:
        safe_patch["electronicPropsMP"] = mp_data["electronicPropsMP"]

    doc_ref.set(safe_patch, merge=True)

    return {
        "formula": formula,
        "status": "ok",
        "mpId": mp_data["mpId"],
        "resolution": resolution,
        "bandgapEv": mp_data.get("electronicPropsMP", {}).get("bandgapEv"),
    }


DEFAULT_FORMULAS = list(CURATED_MP_IDS.keys())


def sync_batch(
    formulas: list[str],
    api_key: str,
    db: Any,
) -> list[dict[str, Any]]:
    results = []
    for formula in formulas:
        try:
            r = sync_to_firestore(formula, api_key, db)
            results.append(r)
            logger.info("Synced %s: %s (%s)", formula, r["status"], r.get("resolution"))
        except Exception as exc:
            logger.exception("Sync failed for %s", formula)
            results.append({"formula": formula, "status": "error", "error": str(exc)[:200]})
    return results
