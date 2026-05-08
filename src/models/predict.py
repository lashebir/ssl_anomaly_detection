#!/usr/bin/env python3
"""
XGBoost Phishing Detection - Prediction Script

Use trained XGBoost model to predict phishing probability for new domains.

Usage:
    # Predict on new cert file
    python src/models/predict.py --input sources/raw/certs_new.jsonl --output predictions.csv

    # Predict on single domain
    python src/models/predict.py --domain "paypa1-secure.com"
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.features.feature_engineering import FeatureEngineer


def load_model(model_dir: Path):
    """
    Load trained XGBoost model and feature list.

    Args:
        model_dir: Directory containing model artifacts

    Returns:
        (model, feature_list)
    """
    model_path = model_dir / 'xgboost_phishing_detector.joblib'
    feature_list_path = model_dir / 'feature_list.json'

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not feature_list_path.exists():
        raise FileNotFoundError(f"Feature list not found: {feature_list_path}")

    model = joblib.load(model_path)

    with open(feature_list_path, 'r') as f:
        feature_list = json.load(f)

    return model, feature_list


def predict_from_file(
    input_path: Path,
    model_dir: Path,
    output_path: Path = None,
    threshold: float = 0.5
):
    """
    Predict phishing probability for certificates in JSONL file.

    Args:
        input_path: Path to input JSONL file with certificates
        model_dir: Directory containing model artifacts
        output_path: Path to save predictions (CSV)
        threshold: Classification threshold (default: 0.5)
    """
    print("="*80)
    print("PHISHING DETECTION - BATCH PREDICTION")
    print("="*80)

    # Load model
    print(f"\n1. Loading model from {model_dir}...")
    model, feature_list = load_model(model_dir)
    print(f"   ✅ Model loaded")
    print(f"   Features: {len(feature_list)}")

    # Load certificates
    print(f"\n2. Loading certificates from {input_path}...")
    df_certs = pd.read_json(input_path, lines=True)
    print(f"   ✅ Loaded {len(df_certs):,} certificates")

    # Extract features
    print(f"\n3. Extracting features...")
    fe = FeatureEngineer(use_whois=False, use_abuseipdb=False, use_greynoise=False)
    df_features = fe.extract_features(df_certs, explode_domains=True, verbose=True)

    # Prepare feature matrix
    X = df_features[feature_list].fillna(0)

    # Predict
    print(f"\n4. Running predictions...")
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Add predictions to dataframe
    df_features['phishing_probability'] = y_pred_proba
    df_features['predicted_label'] = y_pred
    df_features['prediction'] = df_features['predicted_label'].map({
        0: 'legitimate',
        1: 'phishing'
    })

    # Summary
    print(f"\n✅ Predictions complete!")
    print(f"\n   Total domains: {len(df_features):,}")
    print(f"   Predicted phishing: {(y_pred == 1).sum():,} ({(y_pred == 1).mean()*100:.2f}%)")
    print(f"   Predicted legitimate: {(y_pred == 0).sum():,} ({(y_pred == 0).mean()*100:.2f}%)")

    # Show top phishing predictions
    print(f"\n   Top 10 phishing predictions:")
    top_phishing = df_features.nlargest(10, 'phishing_probability')
    for idx, row in top_phishing.iterrows():
        print(f"     - {row['domain']:50s} | prob={row['phishing_probability']:.4f}")

    # Save predictions
    if output_path:
        output_cols = [
            'domain', 'phishing_probability', 'prediction',
            'data_source', 'cert_age_days', 'validity_days'
        ]
        # Add any other columns that exist
        output_cols = [col for col in output_cols if col in df_features.columns]

        df_features[output_cols + feature_list].to_csv(output_path, index=False)
        print(f"\n✅ Predictions saved to: {output_path}")

    return df_features


def predict_single_domain(
    domain: str,
    model_dir: Path
):
    """
    Predict phishing probability for a single domain.

    Note: This creates a minimal certificate record for the domain.
          For real predictions, use actual certificate data.

    Args:
        domain: Domain name to check
        model_dir: Directory containing model artifacts
    """
    print("="*80)
    print("PHISHING DETECTION - SINGLE DOMAIN")
    print("="*80)

    # Load model
    print(f"\n1. Loading model from {model_dir}...")
    model, feature_list = load_model(model_dir)
    print(f"   ✅ Model loaded")

    # Extract features from domain (minimal cert record)
    print(f"\n2. Extracting features for: {domain}")
    fe = FeatureEngineer(use_whois=False, use_abuseipdb=False, use_greynoise=False)

    # Create minimal certificate record
    import time
    now = int(time.time())
    cert_record = pd.DataFrame([{
        'domains': [domain],
        'not_before': now,
        'not_after': now + (90 * 86400),  # 90 days validity
        'issuer': {'CN': 'Unknown', 'O': 'Unknown', 'C': None},
        'subject': {'CN': domain, 'O': None, 'C': None},
        'fingerprint': '00:00:00:00',
        'serial': 0
    }])

    # Extract features
    df_features = fe.extract_features(cert_record, explode_domains=True, verbose=False)

    # Prepare feature matrix
    X = df_features[feature_list].fillna(0)

    # Predict
    print(f"\n3. Running prediction...")
    y_pred_proba = model.predict_proba(X)[0, 1]
    y_pred = int(y_pred_proba >= 0.5)
    prediction = 'PHISHING' if y_pred == 1 else 'LEGITIMATE'

    # Display result
    print("\n" + "="*80)
    print("PREDICTION RESULT")
    print("="*80)
    print(f"\nDomain: {domain}")
    print(f"Prediction: {prediction}")
    print(f"Phishing Probability: {y_pred_proba:.4f} ({y_pred_proba*100:.2f}%)")

    if y_pred_proba >= 0.8:
        print("\n⚠️  HIGH RISK - Very likely phishing")
    elif y_pred_proba >= 0.5:
        print("\n⚠️  MODERATE RISK - Possibly phishing")
    elif y_pred_proba >= 0.2:
        print("\n⚠️  LOW RISK - Some suspicious characteristics")
    else:
        print("\n✅ VERY LOW RISK - Appears legitimate")

    # Show feature values
    print(f"\nTop 5 most important features:")
    feature_importance = model.feature_importances_
    top_indices = np.argsort(feature_importance)[::-1][:5]

    for i, idx in enumerate(top_indices, 1):
        feat_name = feature_list[idx]
        feat_value = X[feat_name].iloc[0]
        importance = feature_importance[idx]
        print(f"  {i}. {feat_name:30s} = {feat_value:8.3f} (importance: {importance:.3f})")

    print("\n" + "="*80)

    return {
        'domain': domain,
        'prediction': prediction,
        'probability': float(y_pred_proba),
        'features': df_features[feature_list].iloc[0].to_dict()
    }


def main():
    parser = argparse.ArgumentParser(description='Predict phishing probability using trained XGBoost model')
    parser.add_argument('--model-dir', type=Path,
                       default=Path('models'),
                       help='Directory containing trained model')
    parser.add_argument('--input', type=Path,
                       help='Input JSONL file with certificates')
    parser.add_argument('--output', type=Path,
                       help='Output CSV file for predictions')
    parser.add_argument('--domain', type=str,
                       help='Single domain to check')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold (default: 0.5)')

    args = parser.parse_args()

    if args.domain:
        # Single domain prediction
        predict_single_domain(args.domain, args.model_dir)
    elif args.input:
        # Batch prediction
        predict_from_file(args.input, args.model_dir, args.output, args.threshold)
    else:
        parser.error("Must provide either --input or --domain")


if __name__ == "__main__":
    main()
