#!/usr/bin/env python3
"""
Add model predictions (y_proba) to features.parquet for dashboard visualization.

This script:
1. Loads features.parquet
2. Loads trained XGBoost model
3. Computes y_proba (predicted probability) for all domains
4. Adds y_proba column to features.parquet

Usage:
    python scripts/add_predictions_to_features.py
    python scripts/add_predictions_to_features.py --features custom_features.parquet --model custom_model.pkl
"""

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    import joblib
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    print("\nInstall dependencies:")
    print("  pip install pandas numpy joblib")
    sys.exit(1)


def add_predictions(features_file: str, model_file: str, output_file: str = None):
    """Add y_proba predictions to features file"""

    print("=" * 80)
    print("  Adding Model Predictions to Features")
    print("=" * 80)

    # Check inputs
    if not Path(features_file).exists():
        print(f"\n❌ ERROR: Features file not found: {features_file}")
        sys.exit(1)

    if not Path(model_file).exists():
        print(f"\n❌ ERROR: Model file not found: {model_file}")
        sys.exit(1)

    if output_file is None:
        output_file = features_file

    print(f"\nInput:  {features_file}")
    print(f"Model:  {model_file}")
    print(f"Output: {output_file}")

    # Load features
    print("\n1. Loading features...")
    df = pd.read_parquet(features_file)
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")

    # Load model
    print("\n2. Loading model...")
    model_data = joblib.load(model_file)
    model = model_data['model']
    feature_cols = model_data['feature_cols']

    print(f"   Model trained: {model_data['trained_at']}")
    print(f"   Recall: {model_data['metrics']['recall']:.4f}")
    print(f"   Features: {len(feature_cols)}")

    # Check if all features exist
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        print(f"\n❌ ERROR: Missing features in data: {missing}")
        sys.exit(1)

    print(f"   ✅ All {len(feature_cols)} features present")

    # Prepare feature matrix
    print("\n3. Preparing features for prediction...")
    X = df[feature_cols].copy()

    # Handle missing/infinite values
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    # Compute predictions
    print("\n4. Computing predictions (this may take a while for large datasets)...")

    # Predict in batches to avoid memory issues
    batch_size = 50000
    y_proba_list = []

    for i in range(0, len(X), batch_size):
        batch = X.iloc[i:i+batch_size]
        y_proba_batch = model.predict_proba(batch)[:, 1]  # Probability of class 1 (phishing)
        y_proba_list.append(y_proba_batch)

        if (i // batch_size + 1) % 10 == 0 or i + batch_size >= len(X):
            print(f"   Processed {min(i + batch_size, len(X)):,} / {len(X):,} rows...")

    y_proba = np.concatenate(y_proba_list)

    # Add to dataframe
    df['y_proba'] = y_proba

    print(f"   ✅ Predictions complete!")

    # Show prediction statistics
    print("\n5. Prediction Statistics:")
    print(f"   Mean y_proba: {y_proba.mean():.4f}")
    print(f"   Median y_proba: {np.median(y_proba):.4f}")
    print(f"   Min y_proba: {y_proba.min():.4f}")
    print(f"   Max y_proba: {y_proba.max():.4f}")

    # Show predictions by actual label
    if 'y' in df.columns:
        print("\n6. Predictions by Actual Label:")
        for label, name in [(0, "Legitimate"), (1, "Phishing")]:
            subset = df[df['y'] == label]['y_proba']
            if len(subset) > 0:
                print(f"   {name:12s}: mean={subset.mean():.4f}, median={subset.median():.4f}")

    # Show high-confidence predictions
    print("\n7. High-confidence predictions (y_proba > 0.9):")
    high_conf = (df['y_proba'] > 0.9).sum()
    print(f"   Count: {high_conf:,} ({high_conf/len(df):.2%})")

    if high_conf > 0 and 'domain' in df.columns:
        print("   Sample domains:")
        samples = df[df['y_proba'] > 0.9]['domain'].head(5)
        for domain in samples:
            print(f"     - {domain}")

    # Save output
    print(f"\n8. Saving to {output_file}...")
    df.to_parquet(output_file)

    size_mb = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"   ✅ Saved! ({size_mb:.1f} MB)")

    print("\n" + "=" * 80)
    print("  ✅ Predictions Added Successfully!")
    print("=" * 80)
    print("\nYou can now use the Operating Threshold section in the dashboard.")
    print("The dashboard will show:")
    print("  - Precision-Recall curve")
    print("  - Optimal threshold recommendations")
    print("  - Trade-offs between precision and recall")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Add model predictions to features file for dashboard"
    )
    parser.add_argument(
        "--features",
        default="features.parquet",
        help="Path to features parquet file (default: features.parquet)"
    )
    parser.add_argument(
        "--model",
        default="src/models/xgb_model_latest.pkl",
        help="Path to trained model (default: src/models/xgb_model_latest.pkl)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: overwrites input features file)"
    )

    args = parser.parse_args()

    add_predictions(args.features, args.model, args.output)


if __name__ == "__main__":
    main()
