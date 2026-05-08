#!/usr/bin/env python3
"""
Example: Complete XGBoost Training and Prediction Pipeline

This demonstrates the full workflow:
1. Load data
2. Extract features
3. Train XGBoost model
4. Make predictions
5. Evaluate results

Run this after you have:
- Collected live certs (src/data/ingest_certificates_labels.py live-certs)
- Collected historical certs (src/data/ingest_certificates_labels.py historical-phishing-certs)
- Run EDA notebook to generate feature statistics
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*80)
print("XGBOOST PHISHING DETECTION - COMPLETE EXAMPLE")
print("="*80)

# Step 1: Check prerequisites
print("\n1. Checking prerequisites...")

live_certs_path = Path("sources/raw/certs_live.jsonl")
hist_certs_path = Path("sources/raw/certs_phishing_historical.jsonl")
feature_stats_path = Path("sources/processed/comprehensive_feature_stats.csv")

if not live_certs_path.exists():
    print(f"   ❌ Missing: {live_certs_path}")
    print("   Run: python src/data/ingest_certificates_labels.py live-certs --duration 300")
    sys.exit(1)

if not hist_certs_path.exists():
    print(f"   ❌ Missing: {hist_certs_path}")
    print("   Run: python src/data/ingest_certificates_labels.py historical-phishing-certs")
    sys.exit(1)

if not feature_stats_path.exists():
    print(f"   ❌ Missing: {feature_stats_path}")
    print("   Run: jupyter notebook notebooks/eda_sandbox.ipynb (run all cells)")
    sys.exit(1)

print("   ✅ All prerequisites found")

# Step 2: Train model
print("\n2. Training XGBoost model...")
print("   (This may take a few minutes...)")

import subprocess
result = subprocess.run([
    sys.executable,
    "src/models/train_xgboost.py",
    "--top-n", "10",
    "--test-size", "0.2"
], capture_output=False)

if result.returncode != 0:
    print("   ❌ Training failed")
    sys.exit(1)

print("\n   ✅ Training complete!")

# Step 3: Make predictions on test domain
print("\n3. Testing prediction on known phishing domain...")

result = subprocess.run([
    sys.executable,
    "src/models/predict.py",
    "--domain", "paypa1-secure.xyz"
], capture_output=False)

print("\n4. Testing prediction on known legitimate domain...")

result = subprocess.run([
    sys.executable,
    "src/models/predict.py",
    "--domain", "google.com"
], capture_output=False)

# Step 4: Summary
print("\n" + "="*80)
print("EXAMPLE COMPLETE!")
print("="*80)

print("\nYou've successfully:")
print("  ✅ Trained an XGBoost phishing detection model")
print("  ✅ Made predictions on test domains")
print("  ✅ Generated evaluation visualizations")

print("\nNext steps:")
print("  1. Check model performance: models/metrics.json")
print("  2. Review visualizations: models/*.png")
print("  3. Make batch predictions:")
print("     python src/models/predict.py --input <cert_file> --output predictions.csv")
print("  4. Integrate with live ingestion pipeline")

print("\nModel artifacts:")
print(f"  - models/xgboost_phishing_detector.joblib")
print(f"  - models/feature_list.json")
print(f"  - models/metrics.json")
print(f"  - models/confusion_matrix.png")
print(f"  - models/roc_curve.png")
print(f"  - models/precision_recall_curve.png")
print(f"  - models/feature_importance.png")
