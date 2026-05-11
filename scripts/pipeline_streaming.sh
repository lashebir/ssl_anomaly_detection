#!/bin/bash
#
# End-to-End Streaming Pipeline for Dashboard
#
# This pipeline:
# 1. Loads streaming data from stream_48_hours_bg.sh into DuckDB
# 2. Runs dbt models to generate features
# 3. Exports features to parquet
# 4. Trains XGBoost model (optional, use --skip-training to use existing)
# 5. Adds y_proba predictions for dashboard PR curves
#
# Usage:
#   ./scripts/pipeline_streaming.sh                    # Full pipeline with training
#   ./scripts/pipeline_streaming.sh --skip-training    # Use existing model
#   ./scripts/pipeline_streaming.sh --incremental      # Only process new data
#
# This script can be run repeatedly as new streaming data arrives.
#

set -e

cd "$(dirname "$0")/.."

# ── Configuration ─────────────────────────────────────────────────────────────

STREAMING_CERTS="sources/raw/certs_streaming_48h.jsonl"
STREAMING_LABELS="sources/raw/labels_streaming_48h.jsonl"
DUCKDB_PATH="feature_store.duckdb"
FEATURES_OUTPUT="features.parquet"
MODEL_OUTPUT="src/models/xgb_model_latest.pkl"

SKIP_TRAINING=false
INCREMENTAL=false

# ── Parse arguments ───────────────────────────────────────────────────────────

for arg in "$@"; do
    case $arg in
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --incremental)
            INCREMENTAL=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-training    Skip model training, use existing model"
            echo "  --incremental      Only process new records (faster for updates)"
            echo "  --help             Show this help message"
            echo ""
            exit 0
            ;;
    esac
done

# ── Header ────────────────────────────────────────────────────────────────────

echo "════════════════════════════════════════════════════════════════════════════"
echo "  Streaming Pipeline for Dashboard"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "  Input (streaming data):"
echo "    Certs:  ${STREAMING_CERTS}"
echo "    Labels: ${STREAMING_LABELS}"
echo ""
echo "  Output:"
echo "    DuckDB:   ${DUCKDB_PATH}"
echo "    Features: ${FEATURES_OUTPUT}"
echo "    Model:    ${MODEL_OUTPUT}"
echo ""
echo "  Mode:"
if [ "$SKIP_TRAINING" = true ]; then
    echo "    Training: SKIPPED (using existing model)"
else
    echo "    Training: ENABLED"
fi
if [ "$INCREMENTAL" = true ]; then
    echo "    Update:   INCREMENTAL (append only)"
else
    echo "    Update:   FULL REBUILD"
fi
echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# ── Check streaming data ──────────────────────────────────────────────────────

if [ ! -f "${STREAMING_CERTS}" ]; then
    echo "❌ ERROR: Streaming certs file not found: ${STREAMING_CERTS}"
    echo ""
    echo "Start streaming collection first:"
    echo "  ./scripts/stream_48_hours_bg.sh"
    echo ""
    echo "Or use a shorter test run:"
    echo "  ./scripts/stream_with_labels.sh"
    echo ""
    exit 1
fi

if [ ! -f "${STREAMING_LABELS}" ]; then
    echo "❌ ERROR: Streaming labels file not found: ${STREAMING_LABELS}"
    echo ""
    echo "Start streaming collection first:"
    echo "  ./scripts/stream_48_hours_bg.sh"
    echo ""
    exit 1
fi

# Show data statistics
CERT_COUNT=$(wc -l < "${STREAMING_CERTS}" | tr -d ' ')
LABEL_COUNT=$(wc -l < "${STREAMING_LABELS}" | tr -d ' ')

echo "✅ Found streaming data:"
echo "   Certificates: ${CERT_COUNT}"
echo "   Labels:       ${LABEL_COUNT}"
echo ""

# ── Step 1: Load into DuckDB ──────────────────────────────────────────────────

echo "Step 1/6: Loading streaming data into DuckDB..."
echo ""

if [ "$INCREMENTAL" = true ] && [ -f "${DUCKDB_PATH}" ]; then
    echo "   Mode: INCREMENTAL (appending new records)"
    # TODO: Implement incremental logic using timestamps
    echo "   ⚠️  Incremental mode not yet implemented, using full rebuild"
    INCREMENTAL=false
fi

if [ "$INCREMENTAL" = false ]; then
    # Full rebuild
    if [ -f "${DUCKDB_PATH}" ]; then
        echo "   Removing existing database for full rebuild..."
        rm -f "${DUCKDB_PATH}"
    fi

    python3 scripts/load_data_to_duckdb.py \
        --db "${DUCKDB_PATH}" \
        --certs "${STREAMING_CERTS}" \
        --labels "${STREAMING_LABELS}"
fi

echo ""

# ── Step 2: Load dbt seeds ────────────────────────────────────────────────────

echo "Step 2/6: Loading dbt seeds (brands, TLDs, keywords)..."
echo ""

dbt seed --profiles-dir .

echo ""

# ── Step 3: Run dbt models ────────────────────────────────────────────────────

echo "Step 3/6: Running dbt models (feature engineering)..."
echo ""

dbt run --profiles-dir .

echo ""

# ── Step 4: Export features ───────────────────────────────────────────────────

echo "Step 4/6: Exporting features to ${FEATURES_OUTPUT}..."
echo ""

python3 -c "
import duckdb

con = duckdb.connect('${DUCKDB_PATH}')

# Export features
con.execute('''
    COPY (SELECT * FROM main_final.features)
    TO '${FEATURES_OUTPUT}' (FORMAT PARQUET)
''')

# Show statistics
row_count = con.execute('SELECT COUNT(*) FROM main_final.features').fetchone()[0]
label_dist = con.execute('''
    SELECT label_source, y, COUNT(*) as count
    FROM main_final.features
    GROUP BY label_source, y
    ORDER BY y DESC, count DESC
''').fetchall()

print(f'✅ Exported {row_count:,} rows to ${FEATURES_OUTPUT}')
print()
print('Label distribution:')
total_phishing = 0
total_legit = 0
for source, y, count in label_dist:
    marker = '🔴' if y == 1 else '🟢'
    print(f'  {marker} {source:15s} y={y}  {count:,}')
    if y == 1:
        total_phishing += count
    else:
        total_legit += count

if total_phishing > 0:
    print(f'')
    print(f'Total: {total_phishing:,} phishing + {total_legit:,} legitimate')
    print(f'Class ratio: {total_legit/total_phishing:.1f}:1')

con.close()
"

echo ""

# ── Step 5: Train model (optional) ────────────────────────────────────────────

if [ "$SKIP_TRAINING" = false ]; then
    echo "Step 5/6: Training XGBoost model..."
    echo ""

    python3 scripts/train_xgboost.py \
        --input "${FEATURES_OUTPUT}" \
        --output "${MODEL_OUTPUT}"

    echo ""
else
    echo "Step 5/6: Skipping model training (using existing model)"

    if [ ! -f "${MODEL_OUTPUT}" ]; then
        echo "   ❌ ERROR: No existing model found at ${MODEL_OUTPUT}"
        echo "   Run without --skip-training to train a new model"
        exit 1
    fi

    echo "   ✅ Using existing model: ${MODEL_OUTPUT}"
    echo ""
fi

# ── Step 6: Add predictions ───────────────────────────────────────────────────

echo "Step 6/6: Adding y_proba predictions for dashboard..."
echo ""

python3 scripts/add_predictions_to_features.py \
    --features "${FEATURES_OUTPUT}" \
    --model "${MODEL_OUTPUT}"

echo ""

# ── Summary ───────────────────────────────────────────────────────────────────

echo "════════════════════════════════════════════════════════════════════════════"
echo "  ✅ Pipeline Complete!"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "  Output files ready:"
echo "    📊 ${FEATURES_OUTPUT} (with y_proba for PR curves)"
echo "    🤖 ${MODEL_OUTPUT}"
echo "    💾 ${DUCKDB_PATH}"
echo ""
echo "  Next steps:"
echo ""
echo "    1. Launch dashboard:"
echo "       streamlit run src/data/dashboard.py"
echo ""
echo "    2. Continue streaming collection in background:"
echo "       ./scripts/stream_48_hours_bg.sh &"
echo ""
echo "    3. Re-run this pipeline as new data arrives:"
echo "       ./scripts/pipeline_streaming.sh --skip-training"
echo ""
echo "  Dashboard features enabled:"
echo "    ✅ Model Results (with PR curve)"
echo "    ✅ Data Explorer"
echo "    ✅ Drift Detection"
echo "    ✅ Live Monitor"
echo ""

# Show last data update time
if [ -f "${STREAMING_CERTS}" ]; then
    LAST_UPDATE=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "${STREAMING_CERTS}" 2>/dev/null || stat -c "%y" "${STREAMING_CERTS}" 2>/dev/null | cut -d'.' -f1)
    echo "  Last data update: ${LAST_UPDATE}"
    echo ""
fi
