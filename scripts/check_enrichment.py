#!/usr/bin/env python3
"""Check whether Contextual Retrieval enrichment actually ran for a tenant.

The enrich step writes a per-chunk `contextualText` (the LLM-generated context
prepended before embedding). If enrichment did NOT run, every chunk's
contextualText is empty/missing and embed/BM25 fall back to raw `text` — which
makes an enrichment-ON eval identical to OFF.

This reads the chunk docs and reports coverage so we can tell, definitively,
whether the ON measurement is valid. READ-ONLY, ~$0 (no external API).

Usage (worker root, venv active, GCP creds):
    python scripts/check_enrichment.py --tenant tenant-dev-001
    python scripts/check_enrichment.py --tenant tenant-dev-001 --sample 3

@phase R253 (diagnostic)
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser(description="Check contextualText coverage (enrichment).")
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--sample", type=int, default=2, help="print N example contextualText snippets")
    ap.add_argument("--limit", type=int, default=0, help="max papers to scan (0 = all)")
    args = ap.parse_args()

    from google.cloud import firestore

    from src.config import get_settings

    settings = get_settings()
    db = firestore.Client(project=settings.gcp_project_id)
    tid = args.tenant

    papers = list(db.collection(f"tenants/{tid}/papers").where("status", "==", "indexed").get())
    if args.limit:
        papers = papers[: args.limit]

    total_chunks = 0
    with_ctx = 0          # contextualText non-empty
    ctx_differs = 0       # contextualText != text (real prepended context)
    per_paper: list[tuple[str, int, int]] = []
    samples: list[str] = []

    for ps in papers:
        title = (ps.to_dict() or {}).get("title", "")[:40]
        p_total = 0
        p_ctx = 0
        for cs in ps.reference.collection("chunks").stream():
            c = cs.to_dict() or {}
            text = (c.get("text") or "").strip()
            ctx = (c.get("contextualText") or "").strip()
            total_chunks += 1
            p_total += 1
            if ctx:
                with_ctx += 1
                p_ctx += 1
                if ctx != text:
                    ctx_differs += 1
                    if len(samples) < args.sample:
                        # show only the prepended part if contextualText starts differently
                        samples.append(f"[{title}] {ctx[:240]}")
        per_paper.append((title or ps.id, p_total, p_ctx))

    print(f"\n=== enrichment check [tenant={tid}] ===")
    print(f"indexed papers scanned : {len(papers)}")
    print(f"total chunks           : {total_chunks}")
    if total_chunks == 0:
        print("NO CHUNKS — nothing indexed.")
        return
    pct_ctx = 100.0 * with_ctx / total_chunks
    pct_diff = 100.0 * ctx_differs / total_chunks
    print(f"chunks w/ contextualText: {with_ctx}  ({pct_ctx:.1f}%)")
    print(f"  ...where ctx != text : {ctx_differs}  ({pct_diff:.1f}%)")
    print()
    if with_ctx == 0:
        print(">>> VERDICT: enrichment did NOT run. Every chunk falls back to raw text.")
        print(">>> The ON eval is measuring the SAME corpus as OFF — Δ=0 is expected.")
        print(">>> Fix: set ENABLE_ENRICHMENT=true, redeploy worker, Re-index, wait for")
        print(">>>      all papers to return to 'indexed', THEN run the ON eval.")
    elif pct_diff < 50:
        print(">>> VERDICT: partial / suspicious — many chunks lack real context.")
        print(">>> Re-index may be incomplete (still processing) or enrichment half-applied.")
    else:
        print(">>> VERDICT: enrichment IS applied. The ON eval is valid;")
        print(">>> a ~0 delta then means Contextual Retrieval gives no lift on this corpus.")

    if samples:
        print("\n--- sample contextualText (prepended context + chunk) ---")
        for s in samples:
            print(f"  {s}\n")

    # papers with zero enriched chunks (pinpoint stragglers)
    zero = [(t, n) for (t, n, c) in per_paper if c == 0 and n > 0]
    if zero and with_ctx > 0:
        print(f"papers with 0 enriched chunks ({len(zero)}):")
        for t, n in zero[:10]:
            print(f"  - {t} ({n} chunks)")


if __name__ == "__main__":
    main()
