#!/bin/bash
# Re-sync curated materials with structure data (R185-7c-1b).
set -e

WORKER_URL="${WORKER_URL:-https://spectra-worker-5xd6gcfx5q-as.a.run.app}"
TOKEN=$(gcloud auth print-identity-token)

FORMULAS=(MoS2 WS2 WO3 TiO2 ZnO Fe2O3 SnO2 MoSe2 WSe2 MoO3 In2O3 Ga2O3 BiVO4 CuO NiO CoO MnO2 V2O5 Nb2O5 Ta2O5)

for formula in "${FORMULAS[@]}"; do
  echo -n "Syncing $formula... "
  resp=$(curl -s -X POST \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"formula\": \"$formula\"}" \
    "$WORKER_URL/materials/sync")
  echo "$resp" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('status'), '|', d.get('resolution', 'n/a'))" 2>/dev/null || echo "$resp"
  sleep 1
done

echo ""
echo "Done. Verify Firestore: materialProfiles/MoS2 should have 'structure' field."
