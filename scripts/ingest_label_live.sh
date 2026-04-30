#!/bin/bash
# orchestrate.sh
# Runs ingestion and labeling pipeline in sequence.
# Edit BATCH_DURATION to control how long to collect certs each run.

set -e

BATCH_DURATION=300   # seconds to run ct_poller per iteration
ITERATIONS=5         # how many rounds to collect before stopping
CERTS_FILE=${1:-certs_fallback.jsonl}

echo "Starting pipeline — $(date -u)"

for i in $(seq 1 $ITERATIONS); do
    echo ""
    echo "── Iteration $i/$ITERATIONS ──────────────────────────────────"

    echo "[1/3] Collecting certs for ${BATCH_DURATION}s ..."
    timeout $BATCH_DURATION python ingest_certs_live.py || true   # timeout exits 124; treat as normal

    echo "[2/3] Refreshing PhishTank labels ..."
    python ingest_labels_live.py

    echo "Iteration $i complete — $(date -u)"
done

echo ""
echo "Live pipeline done. Files written:"
wc -l certs_fallback.jsonl phishtank_labels.jsonl certs_phishtank_historical.jsonl 2>/dev/null || true