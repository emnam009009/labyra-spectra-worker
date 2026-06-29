"""Audit existing duplicate papers in a tenant (READ-ONLY by default).

Groups active papers by:
  • normalized DOI (lowercased, ACS '.sNNN' SI-suffix stripped — so an article
    DOI and its Supporting-Info '.s001' variant from the old bug group together)
  • contentHash (byte-identical re-uploads)

Usage:
  python scripts/audit_duplicate_dois.py --tenant tenant-dev-001         # report only
  python scripts/audit_duplicate_dois.py --tenant tenant-dev-001 --apply # FLAG extras

--apply marks every paper in a group EXCEPT the keeper (earliest 'indexed', else
earliest uploaded) as status='duplicate' + duplicateOf=<keeper>. This only FLAGS
them (badge); it does NOT delete or remove vectors from the KB — do that via the
app delete action once you've reviewed the report.

Requires: pip install google-cloud-firestore ; ADC creds
  (gcloud auth application-default login  OR  GOOGLE_APPLICATION_CREDENTIALS=...)
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict

from google.cloud import firestore  # type: ignore[import-untyped]

_SI_SUFFIX_RE = re.compile(r"\.s\d{1,4}$", re.IGNORECASE)


def _norm_doi(doi: str) -> str:
    d = (doi or "").strip().lower()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
    return _SI_SUFFIX_RE.sub("", d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", required=True, help="tenant id, e.g. tenant-dev-001")
    ap.add_argument("--project", default=None, help="GCP project (else from ADC)")
    ap.add_argument("--apply", action="store_true", help="flag extras as status=duplicate")
    args = ap.parse_args()

    db = firestore.Client(project=args.project) if args.project else firestore.Client()
    col = db.collection(f"tenants/{args.tenant}/papers")

    papers = []
    for snap in col.stream():
        d = snap.to_dict() or {}
        life = d.get("lifecycleStatus")
        if life and life != "active":
            continue
        if d.get("status") == "duplicate":
            continue  # already flagged
        papers.append(
            {
                "id": snap.id,
                "title": d.get("title") or "",
                "doi_raw": d.get("doi") or "",
                "doi": _norm_doi(d.get("doi") or ""),
                "hash": d.get("contentHash") or "",
                "status": d.get("status") or "",
                "uploadedAt": d.get("uploadedAt") or 0,
            }
        )

    by_doi: dict[str, list] = defaultdict(list)
    by_hash: dict[str, list] = defaultdict(list)
    for p in papers:
        if p["doi"]:
            by_doi[p["doi"]].append(p)
        if p["hash"]:
            by_hash[p["hash"]].append(p)

    # union of dup groups, deduped by the exact set of paper ids
    groups: dict[frozenset, tuple[str, str, list]] = {}
    for k, grp in by_doi.items():
        if len(grp) >= 2:
            groups[frozenset(p["id"] for p in grp)] = ("doi", k, grp)
    for k, grp in by_hash.items():
        if len(grp) >= 2:
            ids = frozenset(p["id"] for p in grp)
            if ids not in groups:  # don't double-report a group already caught by doi
                groups[ids] = ("hash", k, grp)

    if not groups:
        print(f"✓ No duplicate groups. {len(papers)} active papers in '{args.tenant}'.")
        return

    print(f"Found {len(groups)} duplicate group(s) in {len(papers)} active papers "
          f"(tenant '{args.tenant}'):\n")
    flagged = 0
    for kind, key, grp in groups.values():
        srt = sorted(grp, key=lambda p: (p["status"] != "indexed", p["uploadedAt"]))
        label = f"{kind}={key}"
        print(f"── {label}  ({len(grp)} papers)")
        for i, p in enumerate(srt):
            tag = "KEEP" if i == 0 else "DUP "
            doi_show = p["doi_raw"] or "-"
            print(f"   [{tag}] {p['id']}  status={p['status'] or '-':12} "
                  f"doi={doi_show:32} {p['title'][:55]}")
        if args.apply:
            keeper = srt[0]["id"]
            for p in srt[1:]:
                col.document(p["id"]).update(
                    {
                        "status": "duplicate",
                        "duplicateOf": keeper,
                        "statusUpdatedAt": firestore.SERVER_TIMESTAMP,
                        "error": f"Duplicate of {keeper} (audit sweep)",
                    }
                )
                flagged += 1
                print(f"      → flagged {p['id']} duplicate of {keeper}")
        print()

    print("DRY-RUN — nothing changed. Re-run with --apply to flag the DUP rows."
          if not args.apply else f"Applied — flagged {flagged} paper(s) as duplicate.")
    if args.apply:
        print("NOTE: flagging only adds a badge. To remove a copy from the KB/RAG, "
              "delete it via the app (which should also clear its vectors).")


if __name__ == "__main__":
    main()
