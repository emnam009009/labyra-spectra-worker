"""Papers processing module (R167-B).

Async pipeline runner cho RAG paper indexing, triggered via Pub/Sub push
from topic 'paper-processing'. Replaces synchronous TS orchestrator in
labyra-app/src/lib/ai/rag/pipeline/* (Vercel timeout blocker, see ADR-018).

Module structure (R167-B build-out):
  types.py        — Pydantic models (PaperJob envelope, PaperDoc, Chunk, OcrResult)
  errors.py       — CancelledError, FatalError, RetryableError
  state.py        — Firestore state machine writers + readers
  pricing.py      — [B1.5] vendor cost constants
  idempotency.py  — [B2] dedup checks (Pub/Sub at-least-once safety)
  ocr.py          — [B2] Mistral OCR step
  chunking.py     — [B2] Sliding window chunker
  embed.py        — [B3] Voyage embedding REST
  index.py        — [B3] Pinecone serverless upsert
  enrich.py       — [B4] LLM contextual enrichment
  metadata.py     — [B4] Title/year/DOI extraction from page 1
  citation.py     — [B5] DOI references → Crossref/OpenAlex
  orchestrator.py — [B6] Pipeline runner, wired vào /papers/process handler

@phase R167-B1
"""
