#!/usr/bin/env python3
"""R243 — backfill: re-verify every paper's self-DOI using R242 logic.

Re-runs verify_self_doi (resolve + title-match, else reverse-lookup by title)
over the EXISTING corpus, reading the title/DOI already stored in Firestore.
It does NOT re-OCR and does NOT call Gemini or the embedder — only Crossref /
OpenAlex (free) — so backfilling the whole library costs ~nothing and needs no
reprocessing by hand.

What it fixes per paper:
  * a wrong DOI (resolves to a different paper) -> corrected via reverse-lookup
  * a missing DOI that a title search can now recover
  * doiVerified that was True only because a wrong-but-valid DOI resolved
    -> downgraded to False (amber) so the UI stops claiming it's confirmed
  * selfDoiSource relabelled to the R242 semantics (verified / reverse-lookup /
    unverified / empty)
When the DOI actually changes, the journal fields are re-resolved from the new
DOI so they stay consistent. Title/authors are left untouched (a full reprocess
owns those; this is a conservative, DOI-focused pass).

SAFETY: dry-run by default — it prints what WOULD change and writes nothing.
Review the output, then re-run with --apply to persist.

Usage (from the worker repo root, with the worker's GCP creds in env):
    python scripts/reverify_doi.py                 # dry-run, all tenants
    python scripts/reverify_doi.py --tenant <tid>  # dry-run, one tenant
    python scripts/reverify_doi.py --limit 5       # dry-run, first 5 papers
    python scripts/reverify_doi.py --apply          # persist the changes

Requires R242 (src/papers/doi_verify.py) present in the repo.

@phase R243
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from typing import Any

# Allow running as a loose script (`python scripts/reverify_doi.py`): Python puts
# the script's own dir (scripts/) on sys.path, not the repo root, so `from src...`
# would fail. Insert the repo root (parent of scripts/) so imports resolve no
# matter the cwd or how the script is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_update(
    data: dict[str, Any],
    verify_fn: Callable[..., Any],
    resolve_journal_fn: Callable[[str], Any],
) -> tuple[dict[str, Any] | None, str]:
    """Pure decision: given a paper doc, return (update_dict, category).

    category in: DOI FIX | RECOVER | UNVERIFY | relabel | book-skip | nochange.
    Returns (None, ...) when nothing needs writing. Injectable fns keep this
    unit-testable without Firestore / network.
    """
    dtype = (data.get("documentType") or "").strip().lower()
    # R244: books/theses are identified by ISBN via Google Books, not a Crossref
    # article DOI — leave them untouched so the backfill never attaches a
    # (possibly wrong-edition) DOI to a book.
    if dtype in ("book", "thesis"):
        return None, "book-skip"

    title = data.get("title") or ""
    authors = data.get("authors") or []
    year = data.get("year")
    cur_doi = (data.get("doi") or "").strip()
    cur_verified = bool(data.get("doiVerified"))
    cur_source = data.get("selfDoiSource") or ""

    v = verify_fn(cur_doi, title, authors, year, dtype)
    new_doi = (v.doi or "").strip()
    new_verified = bool(v.trusted)
    new_source = v.source

    changed_doi = new_doi.lower() != cur_doi.lower()
    if not (changed_doi or new_verified != cur_verified or new_source != cur_source):
        return None, "nochange"

    update: dict[str, Any] = {
        "doi": new_doi,
        "doiVerified": new_verified,
        "selfDoiSource": new_source,
    }

    # Keep journal consistent with a corrected DOI (free Crossref/OpenAlex read).
    if changed_doi and new_doi:
        jr = resolve_journal_fn(new_doi)
        if getattr(jr, "doi_found", False):
            update["journal"] = jr.journal
            update["journalShort"] = jr.journal_short
            update["journalIssn"] = jr.journal_issn
            update["journalSourceId"] = jr.source_id

    if changed_doi and new_doi and not cur_doi:
        cat = "RECOVER"
    elif changed_doi:
        cat = "DOI FIX"
    elif cur_verified and not new_verified:
        cat = "UNVERIFY"
    else:
        cat = "relabel"
    return update, cat


def _iter_papers(db, tenant: str | None):
    """Yield (tenant_id, paper_id, doc_ref, data) for every paper."""
    if tenant:
        col = db.collection(f"tenants/{tenant}/papers")
        for snap in col.stream():
            yield tenant, snap.id, snap.reference, (snap.to_dict() or {})
        return
    for snap in db.collection_group("papers").stream():
        ref = snap.reference
        # tenants/{tid}/papers/{pid} -> parent.parent.id == tid
        tid = ref.parent.parent.id if ref.parent and ref.parent.parent else "?"
        yield tid, snap.id, ref, (snap.to_dict() or {})


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill self-DOI verification (R243).")
    ap.add_argument("--apply", action="store_true", help="persist changes (default: dry-run)")
    ap.add_argument("--tenant", default=None, help="restrict to one tenant id")
    ap.add_argument("--limit", type=int, default=0, help="process at most N papers (0 = all)")
    ap.add_argument("--sleep", type=float, default=0.3, help="seconds between papers (Crossref)")
    args = ap.parse_args()

    from google.cloud import firestore

    from src.config import get_settings
    from src.papers.doi_verify import verify_self_doi
    from src.papers.journal_resolve import resolve_journal_from_doi

    db = firestore.Client(project=get_settings().gcp_project_id)

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"=== R243 self-DOI backfill [{mode}] tenant={args.tenant or 'ALL'} ===\n")

    counts = {
        "scanned": 0,
        "DOI FIX": 0,
        "RECOVER": 0,
        "UNVERIFY": 0,
        "relabel": 0,
        "book-skip": 0,
    }
    for tid, pid, ref, data in _iter_papers(db, args.tenant):
        if args.limit and counts["scanned"] >= args.limit:
            break
        counts["scanned"] += 1
        try:
            update, cat = compute_update(data, verify_self_doi, resolve_journal_from_doi)
        except Exception as exc:  # never let one bad doc abort the run
            print(f"  [ERROR] {tid}/{pid}: {exc}")
            continue
        if cat == "book-skip":
            counts["book-skip"] += 1
            continue
        if update is None:
            continue
        counts[cat] = counts.get(cat, 0) + 1
        old_doi = (data.get("doi") or "").strip() or "(none)"
        title = (data.get("title") or "")[:55]
        if cat in ("DOI FIX", "RECOVER", "UNVERIFY"):
            print(
                f"  [{cat}] {tid}/{pid}\n"
                f"      title : {title}  [type={data.get('documentType') or '?'}]\n"
                f"      doi   : {old_doi}  ->  {update['doi'] or '(none)'}\n"
                f"      verified={data.get('doiVerified')} -> {update['doiVerified']}  "
                f"source={data.get('selfDoiSource')!r} -> {update['selfDoiSource']!r}"
            )
        else:  # relabel — terse
            print(f"  [relabel] {tid}/{pid}: source -> {update['selfDoiSource']}")
        if args.apply:
            ref.update(update)
        time.sleep(max(0.0, args.sleep))

    print("\n--- summary ---")
    for k in ("scanned", "DOI FIX", "RECOVER", "UNVERIFY", "relabel", "book-skip"):
        print(f"  {k:8}: {counts.get(k, 0)}")
    if not args.apply:
        print("\nDRY-RUN — nothing written. Re-run with --apply to persist.")
    else:
        print("\nAPPLIED. Note: references/topic of DOI-FIXED papers still reflect the old")
        print("DOI until a full reprocess; this pass corrects doi + verification + journal.")


if __name__ == "__main__":
    main()
