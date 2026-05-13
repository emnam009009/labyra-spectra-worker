# labyra-spectra-worker

Python worker for spectrum parsing + AI analysis. Part of [Labyra Platform](https://github.com/labyra-platform/labyra-app).

## Architecture

```
labyra-app (Next.js)
    ↓ publish
Pub/Sub topic: spectra-analysis
    ↓ push subscription
Cloud Run service: spectra-worker (this repo)
    ↓
1. Parse SpectrumMetadata from Firestore
2. Download raw file from GCS
3. Run parser (pymatgen / lmfit / custom)
4. Call Anthropic Claude Sonnet 4.6 for interpretation
5. Write AnalysisResult to Firestore
```

See [`labyra-app/docs/labrya-experiment-database-report.md`](https://github.com/labyra-platform/labyra-app/blob/main/docs/labrya-experiment-database-report.md) for full data flow.

## Local dev

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run locally (requires service-account.json + .env)
uvicorn src.main:app --reload --port 8080

# Test webhook locally
curl -X POST http://localhost:8080/pubsub \
  -H "Content-Type: application/json" \
  -d '{"message": {"data": "eyJzcGVjdHJ1bUlkIjoidGVzdCJ9"}}'
```

## Deploy

```bash
# Build + push to Artifact Registry
gcloud builds submit --tag asia-southeast1-docker.pkg.dev/labyra-prod/labyra-docker/spectra-worker:latest

# Deploy to Cloud Run
gcloud run deploy spectra-worker \
  --image asia-southeast1-docker.pkg.dev/labyra-prod/labyra-docker/spectra-worker:latest \
  --region asia-southeast1 \
  --service-account spectra-worker@labyra-prod.iam.gserviceaccount.com \
  --no-allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 540
```

## Roadmap

- [ ] R160-spectra-3a — Scaffold + echo handler
- [ ] R160-spectra-3b — XRD parser end-to-end
- [ ] R160-spectra-3c — UV-Vis + Raman parsers
- [ ] R160-spectra-3d — AI analysis layer
- [ ] R160-spectra-3e — Error handling + retry
