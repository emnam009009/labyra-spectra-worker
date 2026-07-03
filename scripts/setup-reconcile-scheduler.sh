#!/usr/bin/env bash
# Provision the reconcile sweep: a shared secret in Secret Manager + a Cloud
# Scheduler job that calls POST /dft/reconcile-sweep every 5 minutes with the
# secret in the X-Cron-Secret header. This catches stuck/vanished Batch jobs even
# when no user has the workflow page open (the in-app poller only runs while the
# page is mounted).
#
# Run once. Re-running is safe: it updates the existing job/secret in place.
#   bash scripts/setup-reconcile-scheduler.sh
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-labyra-app-dev}"
REGION="${REGION:-asia-southeast1}"
SERVICE="${SERVICE:-spectra-worker}"
SCHEDULE="${SCHEDULE:-*/5 * * * *}"        # every 5 minutes
SECRET_NAME="dft-cron-secret"
JOB_NAME="dft-reconcile-sweep"

ok() { printf '\033[32m✓\033[0m %s\n' "$1"; }

# 1) Shared secret in Secret Manager (create if missing; add a version if empty).
if ! gcloud secrets describe "$SECRET_NAME" --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud secrets create "$SECRET_NAME" --replication-policy=automatic --project "$PROJECT_ID"
  ok "created secret $SECRET_NAME"
fi
if [ -z "$(gcloud secrets versions list "$SECRET_NAME" --project "$PROJECT_ID" --format='value(name)' 2>/dev/null | head -1)" ]; then
  openssl rand -hex 32 | gcloud secrets versions add "$SECRET_NAME" --data-file=- --project "$PROJECT_ID"
  ok "added secret value"
fi
SECRET_VALUE="$(gcloud secrets versions access latest --secret="$SECRET_NAME" --project "$PROJECT_ID")"

# 2) Wire the secret into the worker (so the endpoint's DFT_CRON_SECRET matches).
#    NOTE: deploy.sh should also include this in --set-secrets so it survives redeploys:
#      DFT_CRON_SECRET=dft-cron-secret:latest
gcloud run services update "$SERVICE" --region "$REGION" --project "$PROJECT_ID" \
  --update-secrets "DFT_CRON_SECRET=${SECRET_NAME}:latest" >/dev/null
ok "worker updated with DFT_CRON_SECRET"

WORKER_URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --project "$PROJECT_ID" --format='value(status.url)')"

# 3) Cloud Scheduler job (create or update).
ARGS=(
  --project "$PROJECT_ID" --location "$REGION"
  --schedule "$SCHEDULE"
  --uri "${WORKER_URL}/dft/reconcile-sweep"
  --http-method POST
  --headers "X-Cron-Secret=${SECRET_VALUE},Content-Type=application/json"
  --message-body '{}'
  --attempt-deadline 120s
)
if gcloud scheduler jobs describe "$JOB_NAME" --project "$PROJECT_ID" --location "$REGION" >/dev/null 2>&1; then
  gcloud scheduler jobs update http "$JOB_NAME" "${ARGS[@]}"
  ok "updated scheduler job $JOB_NAME"
else
  gcloud scheduler jobs create http "$JOB_NAME" "${ARGS[@]}"
  ok "created scheduler job $JOB_NAME ($SCHEDULE)"
fi

echo
ok "Done. Trigger a test run:"
echo "  gcloud scheduler jobs run $JOB_NAME --project $PROJECT_ID --location $REGION"
echo "  gcloud logging read 'textPayload:dft.reconcile_all' --project $PROJECT_ID --freshness=10m --limit 5 --format='value(textPayload)'"
