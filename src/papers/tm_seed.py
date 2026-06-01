"""Auto-mine bilingual abstracts into Translation Memory (RAT seed).

Vietnamese scientific papers usually carry BOTH an English abstract and a
Vietnamese "Tóm tắt" — free parallel data. We store that pair into the lab's
Translation Memory (Pinecone namespace tm__<tenantId>) so future EN->VI
translations of similar passages reuse the lab's own terminology (RAT, ADR-045
Tier 4) — no user-maintained glossary needed.

Mirrors src/lib/ai/rag/translation-memory.ts (tmStore) + vector-store/pinecone.ts
(tm__ namespace, {source, translation, lang} metadata, sha256 id) EXACTLY, and
uses the same embedding (voyage-3-large, input_type=document, 1024-dim), so the
vectors live in the same space and the app's tmRetrieve (query embedding) finds
them. Best-effort + non-fatal.

@phase R226
"""
from __future__ import annotations

import hashlib
import logging

from src.papers.embed import _voyage_embed_batch
from src.papers.index import _pinecone_index

logger = logging.getLogger(__name__)

# Mirror translation-memory.ts constants exactly.
_MIN_LEN = 40  # don't memorize trivially short snippets
_STORE_CAP = 280  # cap stored example text (metadata stays small)
_EMBED_CAP = 2000  # cap text sent to the embedder


def _tm_id(source: str, lang: str) -> str:
    """Mirror tmId() in translation-memory.ts: sha256(lang \\u0000 source)[:40]."""
    return hashlib.sha256(f"{lang}\u0000{source}".encode()).hexdigest()[:40]


def seed_tm_from_abstracts(tenant_id: str, en_abstract: str, vi_abstract: str) -> None:
    """Store the EN->VI abstract pair into the lab's TM. Best-effort + non-fatal.

    Skips unless BOTH abstracts are present and the English source is long enough.
    """
    source = (en_abstract or "").strip()
    target = (vi_abstract or "").strip()
    if len(source) < _MIN_LEN or not target:
        return
    try:
        embeddings, _tokens = _voyage_embed_batch([source[:_EMBED_CAP]])
        if not embeddings or not embeddings[0]:
            logger.warning("tm_seed_embed_empty tenant=%s", tenant_id)
            return
        vector = {
            "id": _tm_id(source, "vi"),
            "values": embeddings[0],
            "metadata": {
                "source": source[:_STORE_CAP],
                "translation": target[:_STORE_CAP],
                "lang": "vi",
            },
        }
        _pinecone_index().upsert(vectors=[vector], namespace=f"tm__{tenant_id}")
        logger.info(
            "tm_seed_abstract tenant=%s src_chars=%d tgt_chars=%d",
            tenant_id, len(source), len(target),
        )
    except Exception as exc:
        logger.warning("tm_seed_failed tenant=%s err=%s", tenant_id, exc)
