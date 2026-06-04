#!/usr/bin/env python3
"""R246 Phase 1 — generate a synthetic golden set for RETRIEVAL evaluation.

To measure whether Contextual Retrieval actually improves chunk retrieval, we
need (query -> the chunk that should be retrieved) pairs. This builds them
automatically so you don't hand-author them:

  1. Sample chunks spread across the indexed papers (skip too-short and
     reference-list chunks — questions from those are noise).
  2. For each, Gemini writes ONE natural question answerable from that chunk,
     PARAPHRASED (not copying the chunk's distinctive phrases/numbers) so the
     eval tests semantic retrieval — exactly the case Contextual Retrieval is
     meant to improve — rather than trivial keyword overlap.
  3. The ground-truth target is that chunk: chunkId = f"{paperId}-{chunkIdx}".

The set is built from RAW chunk text (not contextualText) so it is identical
whether enrichment is on or off — the same fixed yardstick for the A/B.

Written to Firestore tenants/{tid}/_evalRetrieval/golden (a separate collection
from _evals so it doesn't pollute the weekly-eval view). The Phase 2 app route
reads it, runs the real retrieval per query, and reports recall@K / MRR.

SAFETY: dry-run by default (prints the pairs to eyeball); --apply writes.

Usage (worker repo root, venv active, GCP + Gemini creds):
    python scripts/gen_retrieval_goldenset.py --tenant tenant-dev-001
    python scripts/gen_retrieval_goldenset.py --tenant tenant-dev-001 --n 50 --apply

@phase R246
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GEN_SYSTEM = (
    "You are building a retrieval-evaluation set for a scientific-paper search "
    "system. Given an excerpt from a paper, write ONE specific question a "
    "researcher would realistically ask whose answer is contained in this "
    "excerpt. Rules: (a) answerable from the excerpt alone; (b) do NOT copy "
    "distinctive phrases or numbers verbatim — paraphrase, so it tests meaning "
    "not keyword overlap; (c) specific enough to have a unique answer (avoid "
    "'what is this about'); (d) one sentence; (e) output ONLY the question.\n"
    "If the excerpt is NOT substantive scientific content — e.g. a reference / "
    "citation list, author names or affiliations, acknowledgements, funding, "
    "keywords, table of contents, or news/editorial matter — output exactly "
    "SKIP and nothing else (do not invent a bibliographic question)."
)

# Sections whose chunks make poor eval targets (a question from a reference
# list or acknowledgements is noise, not a content-retrieval test).
_SKIP_SECTION_RE = re.compile(
    r"\b(reference|bibliograph|acknowledg|author contribution|author inform|"
    r"additional inform|conflict|competing interest|funding|data availab|"
    r"keyword|news|editor|highlight|abbreviation|nomenclature|table of content|"
    r"supporting information|supplementary)",
    re.IGNORECASE,
)


def clean_question(raw: str) -> str:
    """Normalise the model's output to a single bare question line."""
    q = (raw or "").strip()
    # take first non-empty line
    for line in q.splitlines():
        line = line.strip()
        if line:
            q = line
            break
    # drop a leading "Question:" / numbering / quotes
    q = re.sub(r"^\s*(question\s*[:\-]\s*|\d+[.)]\s*)", "", q, flags=re.IGNORECASE)
    q = q.strip().strip('"').strip()
    return q


def is_usable_chunk(text: str, section: str, min_chars: int) -> bool:
    if len((text or "").strip()) < min_chars:
        return False
    return not (section and bool(_SKIP_SECTION_RE.search(section)))


def select_chunks(
    papers: list[dict],
    n: int,
    per_paper: int,
    min_chars: int,
    seed: int = 42,
) -> list[dict]:
    """Pure: pick up to n usable chunks, spread round-robin across papers.

    papers: [{id, title, chunks: [{chunkIdx, text, section}]}]
    Returns: [{paperId, title, chunkIdx, text, section}]
    """
    rng = random.Random(seed)
    # per-paper shuffled pools of usable chunks
    pools: list[tuple[dict, list[dict]]] = []
    for p in papers:
        usable = [c for c in p.get("chunks", []) if is_usable_chunk(c.get("text", ""), c.get("section", ""), min_chars)]
        if usable:
            rng.shuffle(usable)
            pools.append((p, usable))
    rng.shuffle(pools)

    picked: list[dict] = []
    taken: dict[str, int] = {}
    # round-robin passes until n reached or pools exhausted
    progress = True
    while len(picked) < n and progress:
        progress = False
        for p, usable in pools:
            if len(picked) >= n:
                break
            if taken.get(p["id"], 0) >= per_paper:
                continue
            if not usable:
                continue
            c = usable.pop()
            taken[p["id"]] = taken.get(p["id"], 0) + 1
            picked.append({
                "paperId": p["id"],
                "title": p.get("title", ""),
                "chunkIdx": c["chunkIdx"],
                "text": c["text"],
                "section": c.get("section", ""),
            })
            progress = True
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate retrieval golden set (R246 P1).")
    ap.add_argument("--tenant", required=True, help="tenant id (golden set is per-tenant)")
    ap.add_argument("--n", type=int, default=50, help="number of query/chunk pairs")
    ap.add_argument("--per-paper", type=int, default=2, help="max chunks sampled per paper")
    ap.add_argument("--min-chars", type=int, default=400, help="skip chunks shorter than this")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--apply", action="store_true", help="write to Firestore (default: dry-run)")
    args = ap.parse_args()

    from google.cloud import firestore

    from src.config import get_settings
    from src.papers._gemini_client import extract_text

    settings = get_settings()
    model = settings.gemini_model_classify  # cheap Flash, fine for Q-gen
    db = firestore.Client(project=settings.gcp_project_id)
    tid = args.tenant

    papers_snap = (
        db.collection(f"tenants/{tid}/papers").where("status", "==", "indexed").get()
    )
    papers: list[dict] = []
    for ps in papers_snap:
        pd = ps.to_dict() or {}
        chunks = [
            {"chunkIdx": c.get("chunkIdx"), "text": c.get("text", ""), "section": c.get("section", "")}
            for c in (cs.to_dict() or {} for cs in ps.reference.collection("chunks").stream())
        ]
        papers.append({"id": ps.id, "title": pd.get("title", ""), "chunks": chunks})

    # Over-sample so LLM SKIPs (non-content chunks) don't shrink the set below --n.
    chosen = select_chunks(papers, int(args.n * 1.5) + 5, args.per_paper, args.min_chars, args.seed)
    print(
        f"=== golden-set gen [tenant={tid}] {'APPLY' if args.apply else 'DRY-RUN'} ===\n"
        f"papers indexed={len(papers)}  usable chunks picked={len(chosen)}  model={model}\n"
    )

    items: list[dict] = []
    for i, c in enumerate(chosen, 1):
        raw, _in, _out = extract_text(
            model=model,
            prompt=f"<excerpt>\n{c['text'][:4000]}\n</excerpt>",
            system_instruction=GEN_SYSTEM,
            max_tokens=120,
            temperature=0.4,
        )
        q = clean_question(raw)
        if q.upper().startswith("SKIP"):
            print(f"  [skip] {c['paperId']}-{c['chunkIdx']}: non-content (LLM SKIP)")
            continue
        if len(q) < 12:  # model failed / too vague
            print(f"  [skip] {c['paperId']}-{c['chunkIdx']}: empty/short question")
            continue
        item = {
            "id": f"{c['paperId']}-{c['chunkIdx']}",
            "query": q,
            "paperId": c["paperId"],
            "chunkIdx": c["chunkIdx"],
            "section": c["section"],
            "chunkPreview": c["text"][:200],
        }
        items.append(item)
        print(f"  [{i:2}] {c['title'][:32]:32} #{c['chunkIdx']:<3} {('('+c['section']+')')[:18]:18}\n        Q: {q}")
        if len(items) >= args.n:
            break

    print(f"\n--- {len(items)} usable pairs ---")
    if args.apply:
        db.document(f"tenants/{tid}/_evalRetrieval/golden").set({
            "schemaVersion": 1,
            "items": items,
            "count": len(items),
            "model": model,
            "seed": args.seed,
            "generatedAt": int(time.time() * 1000),
        })
        print(f"WROTE tenants/{tid}/_evalRetrieval/golden ({len(items)} items)")
    else:
        print("DRY-RUN — nothing written. Eyeball the questions above, then --apply.")


if __name__ == "__main__":
    main()
