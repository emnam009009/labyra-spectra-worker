#!/usr/bin/env bash
# Deploy spectra-worker to Cloud Run. Idempotent: re-creates Pub/Sub subscription
# with current Cloud Run URL each run (handles URL changes after redeploy).
#
# Usage:
#   bash deploy.sh
#
# Pre-requisites: r160-spectra-3-gcp-setup-v2.sh must have run successfully.

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-labyra-app-dev}"
REGION="${REGION:-asia-southeast1}"
SERVICE="${SERVICE:-spectra-worker}"
REPO="${REPO:-labyra-docker}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:$(date +%Y%m%d-%H%M%S)"
SA_EMAIL="spectra-worker@${PROJECT_ID}.iam.gserviceaccount.com"
TOPIC="spectra-analysis"
DLQ="spectra-analysis-dlq"
SUBSCRIPTION="spectra-worker-push"
TOPIC_PAPERS="paper-processing"
DLQ_PAPERS="paper-processing-dlq"
SUBSCRIPTION_PAPERS="spectra-worker-papers-push"
FIREBASE_BUCKET="${FIREBASE_BUCKET:-${PROJECT_ID}.firebasestorage.app}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }
step() { echo -e "\n${GREEN}━━━ $* ━━━${NC}"; }

gcloud config set project "$PROJECT_ID" --quiet

# ---------- Step 1: Build image ---------------------------------------------
step "Step 1/4: Cloud Build"
gcloud builds submit --tag "$IMAGE" --quiet
ok "Image built: $IMAGE"

# ---------- Step 2: Deploy Cloud Run ----------------------------------------
step "Step 2/4: Deploy Cloud Run"
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --region "$REGION" \
  --service-account "$SA_EMAIL" \
  --no-allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 3600 \
  --max-instances 10 \
  --min-instances 0 \
  --concurrency 1 \
  --set-env-vars "GCP_PROJECT_ID=${PROJECT_ID},GCP_REGION=${REGION},FIREBASE_BUCKET=${FIREBASE_BUCKET},DEFAULT_LOCALE=en,ANALYSIS_VERSION=xrd-1.0.0,PINECONE_INDEX_NAME=labyra-papers,OCR_ENGINE=datalab,OCR_FALLBACK=mistral,ENABLE_ENRICHMENT=true" \
  --set-secrets "ANTHROPIC_API_KEY=anthropic-api-key:latest,MP_API_KEY=mp-api-key:latest,MISTRAL_API_KEY=mistral-api-key:latest,VOYAGE_API_KEY=voyage-api-key:latest,PINECONE_API_KEY=pinecone-api-key:latest,GEMINI_API_KEY=gemini-api-key:latest,BOOKS_API_KEY=books-api-key:latest,OPENALEX_API_KEY=openalex-api-key:latest,DATALAB_API_KEY=datalab-api-key:latest" \
  --quiet

SERVICE_URL=$(gcloud run services describe "$SERVICE" --region="$REGION" --format="value(status.url)")
ok "Deployed: $SERVICE_URL"

# ---------- Step 3: Grant Pub/Sub invoker ----------------------------------
step "Step 3/4: Pub/Sub invoker"
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")
PUBSUB_SA="service-${PROJECT_NUMBER}@gcp-sa-pubsub.iam.gserviceaccount.com"
gcloud run services add-iam-policy-binding "$SERVICE" \
  --region="$REGION" \
  --member="serviceAccount:${PUBSUB_SA}" \
  --role="roles/run.invoker" --quiet >/dev/null
ok "Pub/Sub SA can invoke Cloud Run"

# ---------- Step 4: Create / update push subscription ----------------------
step "Step 4/4: Pub/Sub push subscription"
PUSH_ENDPOINT="${SERVICE_URL}/pubsub"

if gcloud pubsub subscriptions describe "$SUBSCRIPTION" >/dev/null 2>&1; then
  warn "Subscription exists, updating push endpoint..."
  gcloud pubsub subscriptions update "$SUBSCRIPTION" \
    --push-endpoint="$PUSH_ENDPOINT" \
    --push-auth-service-account="$SA_EMAIL" \
    --quiet
  ok "Subscription updated"
else
  gcloud pubsub subscriptions create "$SUBSCRIPTION" \
    --topic="$TOPIC" \
    --push-endpoint="$PUSH_ENDPOINT" \
    --push-auth-service-account="$SA_EMAIL" \
    --ack-deadline=540 \
    --message-retention-duration=1d \
    --max-delivery-attempts=3 \
    --dead-letter-topic="$DLQ" \
    --min-retry-delay=10s \
    --max-retry-delay=600s \
    --quiet
  ok "Subscription created with DLQ + retry policy"
fi

echo ""
echo -e "${GREEN}✓ Deploy complete${NC}"
echo ""
echo "Service URL:  $SERVICE_URL"
echo "Push:         $PUSH_ENDPOINT"

# ---------- R167-A: Papers subscription (push endpoint refresh) -------------
step "R167-A: Papers Pub/Sub subscription"

PAPERS_ENDPOINT="${SERVICE_URL}/papers/process"

if gcloud pubsub subscriptions describe "$SUBSCRIPTION_PAPERS" >/dev/null 2>&1; then
  gcloud pubsub subscriptions update "$SUBSCRIPTION_PAPERS" \
    --push-endpoint="$PAPERS_ENDPOINT" \
    --push-auth-service-account="$SA_EMAIL"
  ok "Papers subscription updated → $PAPERS_ENDPOINT"
else
  warn "Papers subscription '$SUBSCRIPTION_PAPERS' missing"
  warn "Run one-time setup first: bash round-167a-1-gcloud-papers-pubsub.sh"
fi

echo "Subscription: $SUBSCRIPTION → $TOPIC (DLQ: $DLQ, max 3 retries)"
echo ""
echo "Test:"
echo "  gcloud pubsub topics publish $TOPIC --message='{\"tenantId\":\"test\",\"spectrumId\":\"test-001\"}'"
echo "  gcloud run services logs read $SERVICE --region=$REGION --limit=20"
