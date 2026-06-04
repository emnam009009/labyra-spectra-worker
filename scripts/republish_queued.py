#!/usr/bin/env python3
"""Re-publish stuck papers to the paper-processing Pub/Sub topic.

When a re-index leaves papers parked at status='queued' (the worker stopped
pulling them), this re-enqueues them so the worker picks them up again. It
publishes the SAME message schema the worker's /papers/process expects
(jobId, tenantId, paperId, version, storagePath, createdBy, enqueuedAt).

THROTTLED: publishes --batch papers then sleeps --sleep seconds, so we don't
re-flood a worker with limited concurrency (which is likely why it stalled).

By default it targets status=='queued' only (the clearly-stuck ones) and is a
DRY-RUN; pass --apply to actually publish.

Usage (worker root, venv active, GCP creds):
    python scripts/republish_queued.py --tenant tenant-dev-001
    python scripts/republish_queued.py --tenant tenant-dev-001 --apply
    python scripts/republish_queued.py --tenant tenant-dev-001 --apply --batch 3 --sleep 45

@phase R256 (recovery)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-publish stuck queued papers to Pub/Sub.")
    ap.add_argument("--tenant", required=True)
    ap.add_argument(
        "--statuses",
        default="queued",
        help="comma-separated statuses to re-publish (default: queued)",
    )
    ap.add_argument("--batch", type=int, default=5, help="papers per batch before sleeping")
    ap.add_argument("--sleep", type=float, default=30.0, help="seconds to sleep between batches")
    ap.add_argument("--topic", default=os.environ.get("PUBSUB_PAPER_TOPIC", "paper-processing"))
    ap.add_argument("--apply", action="store_true", help="actually publish (default: dry-run)")
    args = ap.parse_args()

    from google.cloud import firestore, pubsub_v1

    from src.config import get_settings

    settings = get_settings()
    project = settings.gcp_project_id
    db = firestore.Client(project=project)
    tid = args.tenant
    target = {s.strip() for s in args.statuses.split(",") if s.strip()}

    papers = list(db.collection(f"tenants/{tid}/papers").get())
    stuck = []
    for ps in papers:
        d = ps.to_dict() or {}
        if d.get("status") in target:
            stuck.append((ps.id, d))

    print(f"\n=== republish [tenant={tid}] {'APPLY' if args.apply else 'DRY-RUN'} ===")
    print(f"topic={args.topic}  statuses={sorted(target)}  matched={len(stuck)} papers")
    if not stuck:
        print("Nothing to re-publish.")
        return

    publisher = pubsub_v1.PublisherClient() if args.apply else None
    topic_path = publisher.topic_path(project, args.topic) if publisher else f"projects/{project}/topics/{args.topic}"

    published = 0
    for i, (pid, d) in enumerate(stuck, 1):
        payload = {
            "jobId": str(uuid.uuid4()),
            "tenantId": tid,
            "paperId": pid,
            "version": int(d.get("version", 1) or 1),
            "storagePath": d.get("storagePath", ""),
            "createdBy": d.get("createdBy") or d.get("uploadedBy") or "system",
            "enqueuedAt": int(time.time() * 1000),
        }
        title = (d.get("title") or "")[:36]
        if not payload["storagePath"]:
            print(f"  [skip] {pid} ({title}) — no storagePath")
            continue
        if args.apply and publisher is not None:
            future = publisher.publish(topic_path, json.dumps(payload).encode("utf-8"))
            future.result()  # block until published (confirms it went through)
        print(f"  [{i:3}] {('published' if args.apply else 'would publish')} {pid}  {title}")
        published += 1
        # throttle: pause between batches so the worker can drain
        if args.apply and published % args.batch == 0 and i < len(stuck):
            print(f"  ... sleeping {args.sleep}s (let worker process this batch) ...")
            time.sleep(args.sleep)

    print(f"\n{'PUBLISHED' if args.apply else 'WOULD PUBLISH'} {published} papers to {args.topic}.")
    if args.apply:
        print("Watch progress: python scripts/reindex_progress.py --tenant " + tid + " --no-chunks")
    else:
        print("DRY-RUN — re-run with --apply to publish.")


if __name__ == "__main__":
    main()
