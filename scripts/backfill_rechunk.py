#!/usr/bin/env python3
"""Re-chunk papers indexed before R556, whose chunks cut words in half.

R556 made the chunker snap both window edges to word boundaries. It only affects
papers processed after it: the 118 papers already in the library were chunked
with the old arithmetic and still begin mid-word — "l bandgap energy…" where the
paper says "optical bandgap energy". Every citation drawn from one shows it.

Two modes, and the order matters:

    python scripts/backfill_rechunk.py --dry-run        # count, spend nothing
    python scripts/backfill_rechunk.py --apply          # re-chunk and re-embed

`--dry-run` exists because the decision needs a number, not a feeling. If five
percent of chunks are damaged this is not worth an embedding bill; if sixty
percent are, it plainly is. Run it first and let it argue.

OCR is not repeated: run_ocr_step caches on the PDF's SHA256 (R181), so a
re-chunk reads the cache and pays only for embeddings and the Pinecone write.
That is what makes this affordable at all — and it is why this script goes
through the pipeline rather than reimplementing it.

@phase R559
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from google.cloud import firestore


@dataclass
class Damage:
    """What a paper's stored chunks look like against R556's rules."""

    paper_id: str
    title: str
    chunks: int
    bad_start: int
    bad_end: int

    @property
    def bad(self) -> int:
        """Chunks broken at either edge. A chunk broken at both counts once —
        it is one chunk to rewrite, not two."""
        return max(self.bad_start, self.bad_end)


def _fragment_at_start(text: str, previous: str) -> bool:
    """True when this chunk opens on the tail of a word that the previous chunk
    holds whole.

    There is no text-only way to know that "culations" is a fragment — it looks
    like a word, and no dictionary here would settle it. But the chunks overlap:
    the window steps back OVERLAP_CHARS, so whatever this chunk opens with also
    appears near the end of the one before. If the previous chunk contains a
    longer word *ending* in this chunk's first token, then this token is that
    word's tail and the boundary split it.

    "…improved cal" / "culations involving" → previous ends "cal", this starts
    "culations"; no. The other way round: previous holds "calculations" whole and
    this opens "culations" → fragment, proven, no guessing.

    Four characters minimum: shorter tails coincide with real words too often
    ("the" ends "breathe") and the finding has to be worth acting on.
    """
    if not text or not previous:
        return False
    head = ""
    for ch in text:
        if not ch.isalnum():
            break
        head += ch
    if len(head) < 4:
        return False
    # A whole word in the previous chunk that ends with `head` and is longer.
    for word in previous.split():
        w = "".join(c for c in word if c.isalnum())
        if len(w) > len(head) and w.endswith(head):
            return True
    return False


def _fragment_at_end(text: str, following: str) -> bool:
    """True when this chunk closes on the head of a word the next chunk holds
    whole. The mirror of the test above, and it has to be tested separately:
    R556 snapped both edges because both were broken."""
    if not text or not following:
        return False
    tail = ""
    for ch in reversed(text):
        if not ch.isalnum():
            break
        tail = ch + tail
    if len(tail) < 4:
        return False
    for word in following.split():
        w = "".join(c for c in word if c.isalnum())
        if len(w) > len(tail) and w.startswith(tail):
            return True
    return False


def scan(db: firestore.Client, tenant_id: str) -> list[Damage]:
    """Read every paper's chunks and measure the edges. Reads only."""
    out: list[Damage] = []
    papers = db.collection(f"tenants/{tenant_id}/papers").stream()
    for paper in papers:
        pid = paper.id
        data = paper.to_dict() or {}
        chunks = list(db.collection(f"tenants/{tenant_id}/papers/{pid}/chunks").stream())
        if not chunks:
            continue
        # Ordered by chunkIdx: the test compares each chunk against its
        # neighbours, so the order is part of the evidence, not a nicety.
        texts = [
            ((c.to_dict() or {}).get("text") or "").strip()
            for c in sorted(chunks, key=lambda d: (d.to_dict() or {}).get("chunkIdx", 0))
        ]
        bad_start = 0
        bad_end = 0
        for i, text in enumerate(texts):
            if not text:
                continue
            prev = texts[i - 1] if i > 0 else ""
            nxt = texts[i + 1] if i + 1 < len(texts) else ""
            if _fragment_at_start(text, prev):
                bad_start += 1
            if _fragment_at_end(text, nxt):
                bad_end += 1
        out.append(
            Damage(
                paper_id=pid,
                title=(data.get("title") or "(không tiêu đề)")[:52],
                chunks=len(chunks),
                bad_start=bad_start,
                bad_end=bad_end,
            )
        )
    return out


def report(rows: list[Damage]) -> int:
    """Print the number the decision needs. Returns the damaged-chunk count."""
    if not rows:
        print("Không có paper nào có chunk.")
        return 0

    total_chunks = sum(r.chunks for r in rows)
    total_bad = sum(r.bad for r in rows)
    papers_hit = sum(1 for r in rows if r.bad > 0)
    pct = (total_bad / total_chunks * 100) if total_chunks else 0.0

    print(f"\n{'PAPER':<54} {'CHUNK':>6} {'HỎNG':>6} {'%':>6}")
    print("─" * 76)
    for r in sorted(rows, key=lambda x: x.bad / max(1, x.chunks), reverse=True)[:15]:
        share = r.bad / max(1, r.chunks) * 100
        print(f"{r.title:<54} {r.chunks:>6} {r.bad:>6} {share:>5.0f}%")
    if len(rows) > 15:
        print(f"… và {len(rows) - 15} paper nữa")

    print("─" * 76)
    print(f"{len(rows)} paper · {total_chunks} chunk")
    print(f"{papers_hit} paper có chunk hỏng · {total_bad} chunk hỏng ({pct:.0f}%)")
    print()
    if pct < 10:
        print(f"→ {pct:.0f}% — thấp. Chạy lại có lẽ không đáng tiền embedding.")
    elif pct < 40:
        print(f"→ {pct:.0f}% — ở giữa. Nam quyết.")
    else:
        print(f"→ {pct:.0f}% — cao. Đáng chạy lại.")
    print("\nOCR đã cache theo SHA256 (R181) nên --apply KHÔNG OCR lại;")
    print("chi phí thật là embedding + ghi Pinecone cho số chunk trên.")
    print()
    print("Lưu ý: con số này là CẬN DƯỚI. Phép kiểm chứng minh một mảnh cụt bằng")
    print("cách tìm từ nguyên vẹn trong chunk kề — nên nó bỏ sót mảnh ngắn dưới 4")
    print("ký tự (\"optical\" cắt thành \"l\"). Thà đếm thiếu và chắc, còn hơn đếm")
    print("thừa rồi tiêu tiền theo một con số thổi phồng.")
    return total_bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tenant", default="tenant-dev-001")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="Đếm, không sửa gì")
    g.add_argument("--apply", action="store_true", help="Chunk lại + embed lại")
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Chỉ chạy N paper đầu — chạy thử trước khi chạy cả thư viện",
    )
    args = ap.parse_args()

    db = firestore.Client()
    rows = scan(db, args.tenant)
    bad = report(rows)

    if args.dry_run:
        return 0

    if bad == 0:
        print("Không có gì để chạy lại.")
        return 0

    targets = [r for r in rows if r.bad > 0]
    if args.limit:
        targets = targets[: args.limit]
    print(f"\n--apply: {len(targets)} paper\n")

    ok = 0
    for i, r in enumerate(targets, 1):
        try:
            n_old, n_new = rechunk_one(db, args.tenant, r.paper_id)
            orphan = max(0, n_old - n_new)
            note = f" · {orphan} chunk mồ côi đã xoá" if orphan else ""
            print(f"  [{i}/{len(targets)}] {r.title:<44} {n_old} → {n_new}{note}")
            ok += 1
        except Exception as exc:  # noqa: BLE001 — one bad paper must not stop the run
            print(f"  [{i}/{len(targets)}] {r.title:<44} LỖI: {exc}")

    print(f"\n{ok}/{len(targets)} paper xong.")
    return 0 if ok == len(targets) else 1


def rechunk_one(db: firestore.Client, tenant_id: str, paper_id: str) -> tuple[int, int]:
    """Re-chunk one paper from cached OCR. Returns (old count, new count).

    Order matters, and it is not the obvious one.

    The obvious order is: delete the old chunks, write the new ones. That leaves
    the paper unsearchable in the gap, and if the process dies in that gap it
    leaves it unsearchable for good. So: write first, then remove only what the
    new pass did not overwrite.

    That remainder is the whole hazard here. Chunk ids are `{paper_id}-{idx}`,
    and snapping to word boundaries shortens windows slightly — a paper that was
    19 chunks may come back 15. Upserting overwrites 0..14 and leaves 15..18
    sitting in Firestore *and Pinecone*, still retrievable, still answering
    questions from text that no longer exists in any chunk. Stale chunks are not
    harmless leftovers; they are ghosts with a vote.
    """
    # Real names, checked against the modules rather than remembered. I had
    # written `embed_chunks` and `load_cached_ocr`; neither exists.
    from src.papers.chunking import chunk_paper
    from src.papers.embed import run_embed_step
    from src.papers.index import run_index_step
    from src.papers.ocr import run_ocr_step
    from src.papers.types import PaperDoc

    paper_ref = db.document(f"tenants/{tenant_id}/papers/{paper_id}")
    snap = paper_ref.get()
    if not snap.exists:
        raise RuntimeError("paper không tồn tại")
    # PaperDoc is a Pydantic model, not a dict — run_index_step would have
    # thrown on a raw dict at the first attribute access. extra='ignore' means
    # the app's PROV-O fields pass through harmlessly.
    raw = snap.to_dict() or {}
    paper = PaperDoc(**raw)

    chunks_col = db.collection(f"tenants/{tenant_id}/papers/{paper_id}/chunks")
    old_ids = {c.id for c in chunks_col.stream()}

    storage_path = raw.get("storagePath") or ""
    if not storage_path:
        raise RuntimeError("paper không có storagePath")

    # run_ocr_step checks the SHA256 cache itself (R181) and returns the cached
    # result on a hit. Going through it rather than reaching for the cache
    # directly means this script cannot drift from what ingest actually does —
    # the same reason it calls chunk_paper and run_index_step instead of
    # reimplementing them.
    ocr = run_ocr_step(db, tenant_id, paper_id, storage_path)

    fresh = chunk_paper(ocr)
    if not fresh:
        raise RuntimeError("chunk lại ra 0 chunk")

    embedded = run_embed_step(db, tenant_id, paper_id, fresh)
    run_index_step(db, tenant_id, paper, embedded)

    new_ids = {f"{paper_id}-{c.chunk_idx}" for c in fresh}
    orphans = old_ids - new_ids
    if orphans:
        batch = db.batch()
        for oid in orphans:
            batch.delete(chunks_col.document(oid))
        batch.commit()
        _pinecone_delete(list(orphans))

    return len(old_ids), len(new_ids)


def _pinecone_delete(ids: list[str]) -> None:
    """Drop orphaned vectors. Firestore and Pinecone must agree on what exists —
    a chunk deleted from one and left in the other is worse than either."""
    from src.papers.index import _pinecone_index

    _pinecone_index().delete(ids=ids)


if __name__ == "__main__":
    sys.exit(main())
