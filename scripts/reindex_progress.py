#!/usr/bin/env python3
"""Monitor re-index / enrichment progress for a tenant.

Unlike check_enrichment.py (which only looks at status=='indexed'), this scans
ALL papers and shows the status distribution — so you can watch a re-index move
papers through queued -> ... -> enriching -> ... -> indexed — plus the running
contextualText coverage and enrichment $ spent so far.

Poll it every ~30s. Done when: status all 'indexed' AND contextualText > 0%.

READ-ONLY, ~$0. Usage (worker root, venv active):
    python scripts/reindex_progress.py --tenant tenant-dev-001
    watch -n 30 'python scripts/reindex_progress.py --tenant tenant-dev-001'   # auto-refresh

@phase R255 (diagnostic)
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-index / enrichment progress.")
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--no-chunks", action="store_true", help="skip chunk scan (faster, status only)")
    args = ap.parse_args()

    from google.cloud import firestore

    from src.config import get_settings

    settings = get_settings()
    db = firestore.Client(project=settings.gcp_project_id)
    tid = args.tenant

    papers = list(db.collection(f"tenants/{tid}/papers").get())
    status_counts: Counter[str] = Counter()
    enrich_cost = 0.0
    for ps in papers:
        d = ps.to_dict() or {}
        status_counts[d.get("status", "?")] += 1
        cost = d.get("costUsd")
        if isinstance(cost, dict):
            enrich_cost += float(cost.get("enrichment", 0) or 0)

    total = len(papers)
    indexed = status_counts.get("indexed", 0)

    print(f"\n=== reindex progress [tenant={tid}] ===")
    print(f"papers total : {total}")
    print("status       :")
    for st, n in sorted(status_counts.items(), key=lambda kv: -kv[1]):
        bar = "#" * min(40, n)
        print(f"  {st:12} {n:3}  {bar}")
    print(f"enrichment $ : ${enrich_cost:.3f} (sum costUsd.enrichment across papers)")

    if not args.no_chunks:
        total_chunks = 0
        with_ctx = 0
        for ps in papers:
            for cs in ps.reference.collection("chunks").stream():
                c = cs.to_dict() or {}
                total_chunks += 1
                if (c.get("contextualText") or "").strip():
                    with_ctx += 1
        pct = (100.0 * with_ctx / total_chunks) if total_chunks else 0.0
        print(f"chunks       : {with_ctx}/{total_chunks} have contextualText ({pct:.1f}%)")

    in_flight = total - indexed - status_counts.get("failed", 0)
    print()
    if indexed == total and total > 0:
        print(">>> all papers indexed.")
        if args.no_chunks:
            print(">>> run with chunk scan to confirm contextualText > 0, then run the ON eval.")
        else:
            print(">>> if contextualText > 0% above -> ready: run the ON eval (refit + label on2).")
    elif in_flight > 0:
        print(f">>> STILL PROCESSING: {in_flight} paper(s) not yet indexed. Re-run in ~30s.")
    if status_counts.get("failed", 0):
        print(f">>> WARNING: {status_counts['failed']} failed — check worker logs.")


if __name__ == "__main__":
    main()
