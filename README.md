# labyra-spectra-worker

Python worker for spectrum parsing + AI analysis. Part of [Labyra Platform](https://github.com/labyra-platform/labyra-app).

## Architecture

```
labyra-app (Next.js)
    ↓ publish to Pub/Sub topic 'spectra-analysis'
    ↓ {tenantId, spectrumId, spectrumType}
Cloud Run service 'spectra-worker' (this repo)
    ↓
1. Load SpectrumMetadata from Firestore
2. Transition status: queued → processing
3. Download raw file from GCS
4. Parse (XRD: scipy.find_peaks + Williamson-Hall)
5. Resolve tenant locale (vi/en)
6. Call Anthropic Sonnet 4.6 with hybrid prompt
7. Write AnalysisResult to /tenants/{tid}/spectra/{sid}/analysis/latest
8. Transition status: analyzed
```

## Supported spectrum types (R160-spectra-3a)

- ✅ XRD — peaks + Williamson-Hall + AI phase ID

Planned (spectra-3c, 3d):
- UV-Vis, Raman, FTIR, PL, XPS

## Local dev

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ruff check src tests
mypy src
pytest -v
```

## Deploy

```bash
# Pre-req: r160-spectra-3-gcp-setup-v2.sh has run (GCP infra ready)
bash deploy.sh
```

Deploy script handles:
- Cloud Build (Python image)
- Cloud Run service (private, SA auth)
- Pub/Sub push subscription (creates or updates endpoint)
- DLQ + 3-retry policy

## Test end-to-end

```bash
# Publish test message
gcloud pubsub topics publish spectra-analysis \
  --message='{"tenantId":"YOUR_TENANT","spectrumId":"REAL_SPECTRUM_ID"}'

# Check logs
gcloud run services logs read spectra-worker --region=asia-southeast1 --limit=50
```

## Roadmap

- [x] R160-spectra-3a — Scaffold + XRD parser + AI + Cloud Run
- [ ] R160-spectra-3c — UV-Vis + Raman + FTIR parsers
- [ ] R160-spectra-3d — Frontend AnalysisResult display
- [ ] R160-spectra-3e — Error handling + DLQ inspector UI
- [ ] R160-spectra-3f — EIS (impedance.py) + electrochemistry
