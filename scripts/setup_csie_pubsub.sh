#!/bin/bash
# Setup Pub/Sub for CSIE auto-trigger (R185-8b).
# Topic: csie-trigger
# Subscription: csie-push → worker /csie/process
set -e

PROJECT="${GCP_PROJECT:-labyra-app-dev}"
WORKER_URL="${WORKER_URL:-https://spectra-worker-5xd6gcfx5q-as.a.run.app}"
SERVICE_ACCOUNT="spectra-worker@${PROJECT}.iam.gserviceaccount.com"

echo "Creating Pub/Sub topic csie-trigger..."
gcloud pubsub topics create csie-trigger --project=$PROJECT 2>/dev/null || echo "  already exists"

echo "Creating push subscription csie-push..."
gcloud pubsub subscriptions create csie-push \
    --topic=csie-trigger \
    --push-endpoint="$WORKER_URL/csie/process" \
    --push-auth-service-account="$SERVICE_ACCOUNT" \
    --ack-deadline=60 \
    --message-retention-duration=1h \
    --project=$PROJECT \
    2>/dev/null || echo "  already exists"

echo ""
echo "Verify:"
echo "  gcloud pubsub topics list --project=$PROJECT | grep csie"
echo "  gcloud pubsub subscriptions describe csie-push --project=$PROJECT"
