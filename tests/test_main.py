"""Smoke tests for FastAPI app."""

from __future__ import annotations

import base64
import json

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "labyra-spectra-worker"}


def test_pubsub_echo() -> None:
    payload = {"spectrumId": "test-001", "tenantId": "lab-bku", "spectrumType": "xrd"}
    encoded = base64.b64encode(json.dumps(payload).encode()).decode()
    envelope = {"message": {"data": encoded, "attributes": {}}}

    response = client.post("/pubsub", json=envelope)
    assert response.status_code == 204


def test_pubsub_invalid_envelope() -> None:
    response = client.post("/pubsub", json={"not_a_message": "oops"})
    assert response.status_code == 400
