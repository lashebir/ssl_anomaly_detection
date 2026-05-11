#!/bin/bash
#
# Feature Engineering Pipeline with dbt + DuckDB
#
# This script:
# 1. Loads streaming JSONL data into DuckDB
# 2. Runs dbt models to generate features
# 3. Exports features.parquet for dashboard.py
#
# Usage:
#   ./scripts/run_feature_pipeline.sh
#   ./scripts/run_feature_pipeline.sh --certs sources/raw/test_certs.jsonl --labels sources/raw/test_labels.jsonl
#

set -e

cd "$(dirname "$0")/.."

# Parse arguments
CERTS_FILE="${1:-sources/raw/certs_streaming_48h.jsonl}"
LABELS_FILE="${2:-sources/raw/labels_streaming_48h.jsonl}"
DB_PATH="feature_store.duckdb"
OUTPUT_PARQUET="features.parquet"

echo "═══════════════════════════════════════════════════════════════════════════"
echo "  Feature Engineering Pipeline (dbt + DuckDB)"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "  Input:"
echo "    Certs:  ${CERTS_FILE}"
echo "    Labels: ${LABELS_FILE}"
echo ""
echo "  Output:"
echo "    DuckDB: ${DB_PATH}"
echo "    Parquet: ${OUTPUT_PARQUET}"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

# Step 1: Check if dbt is installed
if ! command -v dbt &> /dev/null; then
    echo "❌ ERROR: dbt not found"
    echo ""
    echo "Install dbt with DuckDB adapter:"
    echo "  pip install dbt-duckdb"
    echo ""
    exit 1
fi

# Step 2: Check if duckdb Python package is installed
if ! python3 -c "import duckdb" 2>/dev/null; then
    echo "❌ ERROR: duckdb Python package not found"
    echo ""
    echo "Install:"
    echo "  pip install duckdb"
    echo ""
    exit 1
fi

# Step 3: Load data into DuckDB
echo "Step 1/4: Loading data into DuckDB..."
echo ""
python3 scripts/load_data_to_duckdb.py \
    --db "${DB_PATH}" \
    --certs "${CERTS_FILE}" \
    --labels "${LABELS_FILE}"

echo ""

# Step 4: Install dbt dependencies (seeds)
echo "Step 2/4: Loading dbt seeds (brands, TLDs, keywords)..."
echo ""
dbt seed --profiles-dir .

echo ""

# Step 5: Run dbt models
echo "Step 3/4: Running dbt models (staging → intermediate → final)..."
echo ""
dbt run --profiles-dir .

echo ""

# Step 6: Export to parquet
echo "Step 4/4: Exporting features to ${OUTPUT_PARQUET}..."
echo ""

python3 -c "
import duckdb

con = duckdb.connect('${DB_PATH}')
con.execute(\"\"\"
    COPY (
        SELECT * FROM final.features
    ) TO '${OUTPUT_PARQUET}' (FORMAT PARQUET)
\"\"\")
con.close()

print('✅ Exported to ${OUTPUT_PARQUET}')
"

# Show summary
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  Pipeline Complete! ✅"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

if [ -f "${OUTPUT_PARQUET}" ]; then
    SIZE=$(du -h "${OUTPUT_PARQUET}" | cut -f1)
    echo "  Output: ${OUTPUT_PARQUET} (${SIZE})"
    echo ""

    # Show feature count
    python3 -c "
import pandas as pd
df = pd.read_parquet('${OUTPUT_PARQUET}')
print(f'  Rows: {len(df):,}')
print(f'  Columns: {len(df.columns)}')
print()
print('  Feature columns:')
feature_cols = [c for c in df.columns if c not in ['fingerprint', 'serial', 'domain', 'timestamp', 'not_before', 'not_after', 'issuer_org', 'tld', 'closest_brand', 'y', 'label_source', 'label_timestamp', 'source', 'data_source']]
for col in sorted(feature_cols):
    print(f'    - {col}')
print()
print('  Label distribution:')
print(df['label_source'].value_counts())
print()
"

    echo ""
    echo "  Next steps:"
    echo "    1. Launch dashboard:"
    echo "       streamlit run src/data/dashboard.py"
    echo ""
    echo "    2. Train models:"
    echo "       jupyter notebook notebooks/model_sandbox.ipynb"
    echo ""
else
    echo "  ❌ ERROR: ${OUTPUT_PARQUET} not created"
    exit 1
fi
