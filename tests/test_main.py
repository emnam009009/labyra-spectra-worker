"""FastAPI smoke tests."""

from __future__ import annotations

import base64
import json

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_healthz() -> None:
    response = client.get("/healthz")
    assert response.status_code == 200


def test_pubsub_missing_envelope() -> None:
    response = client.post("/pubsub", json={"not_a_message": "oops"})
    assert response.status_code == 400


def test_pubsub_missing_tenant_id() -> None:
    payload = {"spectrumId": "test-001"}  # no tenantId
    encoded = base64.b64encode(json.dumps(payload).encode()).decode()
    envelope = {"message": {"data": encoded}}
    response = client.post("/pubsub", json=envelope)
    assert response.status_code == 400
