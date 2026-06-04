#!/usr/bin/env python3
"""R245 — measure reference/citation quality across the corpus (read-only).

Answers the question that gates GROBID (and tells us where the reference layer
is weakest): of every paper's references, how many are RESOLVED against a
registry (metadataSource crossref/openalex — clean title + DOI) versus only
scraped from the PDF (metadataSource "pdf-only" — frequently untitled), and how
many papers have zero or very few references at all.

GROBID's payoff is structuring the reference LIST from PDF layout — exactly the
"pdf-only / untitled" and "too-few-refs" cases. If those are rare, GROBID is not
worth a JVM service; if they dominate, it is.

Reads only Firestore (citations + papers). No Crossref/OpenAlex/OCR calls, no
writes. Cost ~$0.

Usage (from worker repo root, with the worker's GCP creds):
    python scripts/measure_refs.py                 # all tenants
    python scripts/measure_refs.py --tenant <tid>  # one tenant
    python scripts/measure_refs.py --worst 15      # show N worst papers

@phase R245
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

# Run as a loose script: put repo root on sys.path so `from src...` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_NON_ARTICLE = ("book", "thesis")
_KNOWN_SOURCES = ("crossref", "openalex", "pdf-only", "manual")


def summarize(papers: list[dict], citations: list[dict], worst_n: int = 10) -> dict:
    """Pure aggregation — no I/O. Returns a report dict.

    papers: [{id, title, documentType}]
    citations: [{sourcePaperId, metadataSource, targetTitle, targetDoi, lifecycleStatus}]
    """
    active = [c for c in citations if (c.get("lifecycleStatus") or "active") == "active"]
    by_paper: dict[str, list[dict]] = {}
    for c in active:
        by_paper.setdefault(c.get("sourcePaperId") or "", []).append(c)

    articles = [p for p in papers if (p.get("documentType") or "").lower() not in _NON_ARTICLE]
    nonarticles = len(papers) - len(articles)

    src_counts = dict.fromkeys((*_KNOWN_SOURCES, "other"), 0)
    untitled_pdf = 0
    per_paper: list[int] = []
    zero_refs = 0
    low_refs = 0
    pdfonly_heavy = 0
    worst: list[tuple[dict, int, float]] = []

    for p in articles:
        refs = by_paper.get(p["id"], [])
        n = len(refs)
        per_paper.append(n)
        n_pdf = 0
        for c in refs:
            ms = c.get("metadataSource") or "other"
            key = ms if ms in src_counts else "other"
            src_counts[key] += 1
            if ms == "pdf-only":
                n_pdf += 1
                if not (c.get("targetTitle") or "").strip():
                    untitled_pdf += 1
        frac_pdf = (n_pdf / n) if n else 0.0
        if n == 0:
            zero_refs += 1
            worst.append((p, n, 1.0))
        else:
            if n < 5:
                low_refs += 1
            if frac_pdf > 0.5:
                pdfonly_heavy += 1
            if frac_pdf > 0.5:
                worst.append((p, n, frac_pdf))

    total_cit = sum(src_counts.values())
    resolved = src_counts["crossref"] + src_counts["openalex"]
    # worst: 0-ref first, then most pdf-only-heavy
    worst.sort(key=lambda t: (t[1] != 0, -t[2], -t[1]))
    return {
        "articles": len(articles),
        "nonarticles": nonarticles,
        "total_citations": total_cit,
        "src_counts": src_counts,
        "resolved": resolved,
        "untitled_pdf": untitled_pdf,
        "zero_refs": zero_refs,
        "low_refs": low_refs,
        "pdfonly_heavy": pdfonly_heavy,
        "per_paper": per_paper,
        "worst": worst[:worst_n],
    }


def _pct(n: int, d: int) -> str:
    return f"{(100.0 * n / d):.0f}%" if d else "—"


def _print_report(r: dict) -> None:
    print(f"Papers (articles): {r['articles']}    book/thesis (excluded): {r['nonarticles']}")
    print(f"Total active citations: {r['total_citations']}\n")

    print("By metadata source:")
    sc, total = r["src_counts"], r["total_citations"]
    for k in (*_KNOWN_SOURCES, "other"):
        if sc[k]:
            tag = "   <- unresolved / often untitled" if k == "pdf-only" else ""
            print(f"  {k:9}: {sc[k]:5}  ({_pct(sc[k], total)}){tag}")
    print(f"  {'RESOLVED':9}: {r['resolved']:5}  ({_pct(r['resolved'], total)})  [crossref+openalex]\n")

    pp = [n for n in r["per_paper"]]
    if pp:
        med = int(statistics.median(pp))
        print(f"Refs per article: min {min(pp)} / median {med} / max {max(pp)}")
    print(f"Papers with 0 refs           : {r['zero_refs']}    <- extraction found nothing")
    print(f"Papers with <5 refs          : {r['low_refs']}")
    print(f"Papers >50% pdf-only         : {r['pdfonly_heavy']}    <- GROBID would help most")
    print(f"Untitled (pdf-only, no title): {r['untitled_pdf']}\n")

    if r["worst"]:
        print(f"Worst papers (GROBID candidates), top {len(r['worst'])}:")
        for p, n, frac in r["worst"]:
            title = (p.get("title") or "")[:50]
            detail = "0 refs" if n == 0 else f"{n} refs, {frac * 100:.0f}% pdf-only"
            print(f"  {p['id'][:16]}  {title:50}  {detail}")
    print()
    z, h, u = r["zero_refs"], r["pdfonly_heavy"], r["untitled_pdf"]
    if z + h >= max(3, r["articles"] // 5) or u >= r["total_citations"] // 4:
        print("VERDICT HINT: enough 0-ref / pdf-only-heavy / untitled to justify trialing GROBID.")
    else:
        print("VERDICT HINT: reference layer looks healthy — GROBID likely not worth a JVM service yet.")


def _iter_collection(db, tenant: str | None, name: str):
    if tenant:
        for snap in db.collection(f"tenants/{tenant}/{name}").stream():
            yield tenant, snap.id, (snap.to_dict() or {})
        return
    for snap in db.collection_group(name).stream():
        ref = snap.reference
        tid = ref.parent.parent.id if ref.parent and ref.parent.parent else "?"
        yield tid, snap.id, (snap.to_dict() or {})


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure reference quality (R245, read-only).")
    ap.add_argument("--tenant", default=None, help="restrict to one tenant id")
    ap.add_argument("--worst", type=int, default=10, help="how many worst papers to list")
    args = ap.parse_args()

    from google.cloud import firestore

    from src.config import get_settings

    db = firestore.Client(project=get_settings().gcp_project_id)

    papers = [
        {"id": pid, "title": d.get("title"), "documentType": d.get("documentType")}
        for _tid, pid, d in _iter_collection(db, args.tenant, "papers")
    ]
    citations = [
        {
            "sourcePaperId": d.get("sourcePaperId"),
            "metadataSource": d.get("metadataSource"),
            "targetTitle": d.get("targetTitle"),
            "targetDoi": d.get("targetDoi"),
            "lifecycleStatus": d.get("lifecycleStatus"),
        }
        for _tid, _cid, d in _iter_collection(db, args.tenant, "citations")
    ]

    print(f"=== Reference quality [tenant={args.tenant or 'ALL'}] ===")
    print(f"(loaded {len(papers)} papers, {len(citations)} citations)\n")
    _print_report(summarize(papers, citations, args.worst))


if __name__ == "__main__":
    main()
