"""FastAPI entrypoint for Cloud Run. Receives Pub/Sub push messages."""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="labyra-spectra-worker", version="0.1.0")


@app.get("/")
def health() -> dict[str, str]:
    """Health check for Cloud Run."""
    return {"status": "ok", "service": "labyra-spectra-worker"}


@app.post("/pubsub", status_code=status.HTTP_204_NO_CONTENT)
async def handle_pubsub_push(request: Request) -> None:
    """Pub/Sub push subscription endpoint.

    Body format (Pub/Sub envelope):
      { "message": { "data": "<base64-encoded JSON>", "attributes": {...} } }

    For R160-spectra-3a: echo the payload, return 204.
    R160-spectra-3b will dispatch to parser.
    """
    envelope: dict[str, Any] = await request.json()

    if "message" not in envelope:
        raise HTTPException(status_code=400, detail="Invalid Pub/Sub envelope")

    message = envelope["message"]
    encoded_data: str = message.get("data", "")

    if not encoded_data:
        logger.warning("Empty Pub/Sub message data")
        return

    try:
        payload: dict[str, Any] = json.loads(base64.b64decode(encoded_data).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        logger.exception("Failed to decode Pub/Sub message")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info("Received spectrum analysis request: %s", payload)
    # TODO(spectra-3b): dispatch to parser based on payload["spectrumType"]
