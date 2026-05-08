#!/usr/bin/env python3
"""
XGBoost Phishing Detection Model Training

Trains an XGBoost classifier on SSL certificate features to detect phishing domains.
Uses top N features by effect size from comprehensive feature analysis.

Usage:
    python src/models/train_xgboost.py --top-n 10
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)
import xgboost as xgb
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.features.feature_engineering import FeatureEngineer


def load_data(live_path: Path, hist_path: Path, verbose: bool = True):
    """
    Load and combine live + historical certificate data.

    Args:
        live_path: Path to live cert JSONL file
        hist_path: Path to historical phishing cert JSONL file
        verbose: Print progress messages

    Returns:
        Combined dataframe with all features extracted
    """
    if verbose:
        print("="*80)
        print("LOADING DATA")
        print("="*80)

    # Load live data
    if live_path.exists():
        if verbose:
            print(f"\n1. Loading live certificates from {live_path}...")
        df_live_certs = pd.read_json(live_path, lines=True)
        if verbose:
            print(f"   Loaded {len(df_live_certs):,} live certificates")
    else:
        raise FileNotFoundError(f"Live cert file not found: {live_path}")

    # Load historical data
    if hist_path.exists():
        if verbose:
            print(f"\n2. Loading historical phishing certificates from {hist_path}...")
        df_hist_certs = pd.read_json(hist_path, lines=True)
        if verbose:
            print(f"   Loaded {len(df_hist_certs):,} historical phishing certificates")
    else:
        raise FileNotFoundError(f"Historical cert file not found: {hist_path}")

    # Extract features
    if verbose:
        print("\n3. Extracting comprehensive features...")

    fe = FeatureEngineer(use_whois=False, use_abuseipdb=False, use_greynoise=False)

    # Extract features from live data
    df_live = fe.extract_features(df_live_certs, explode_domains=True, verbose=verbose)
    df_live["y"] = 0  # Live data assumed legitimate/unknown
    df_live["data_source"] = "live"

    # Extract features from historical data
    df_hist = fe.extract_features(df_hist_certs, explode_domains=True, verbose=verbose)
    df_hist["y"] = 1  # Historical data is phishing
    df_hist["data_source"] = "historical"

    # Combine datasets
    df_combined = pd.concat([df_live, df_hist], ignore_index=True)

    if verbose:
        print(f"\n✅ Data loaded and features extracted:")
        print(f"   Total samples: {len(df_combined):,}")
        print(f"   Phishing (y=1): {(df_combined['y'] == 1).sum():,} ({(df_combined['y'] == 1).mean()*100:.2f}%)")
        print(f"   Legitimate (y=0): {(df_combined['y'] == 0).sum():,} ({(df_combined['y'] == 0).mean()*100:.2f}%)")

    return df_combined, fe


def get_top_features(stats_path: Path, top_n: int = 10, verbose: bool = True):
    """
    Get top N features by effect size from comprehensive feature statistics.

    Args:
        stats_path: Path to comprehensive_feature_stats.csv
        top_n: Number of top features to select
        verbose: Print progress messages

    Returns:
        List of top feature names
    """
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Feature statistics file not found: {stats_path}\n"
            "Run the EDA notebook (eda_sandbox.ipynb) first to generate feature statistics."
        )

    df_stats = pd.read_csv(stats_path)

    # Sort by absolute effect size
    df_stats = df_stats.sort_values(by='Cohen\'s d', key=lambda x: abs(x), ascending=False)

    # Get top N features
    top_features = df_stats.head(top_n)['Feature'].tolist()

    if verbose:
        print("\n" + "="*80)
        print(f"TOP {top_n} FEATURES BY EFFECT SIZE")
        print("="*80)
        for i, (idx, row) in enumerate(df_stats.head(top_n).iterrows(), 1):
            print(f"{i:2d}. {row['Feature']:30s} | Cohen's d = {row['Cohen\'s d']:6.3f} | "
                  f"p-value = {row['p-value']:.2e}")

    return top_features


def train_model(
    X_train, y_train, X_test, y_test,
    scale_pos_weight: float = None,
    verbose: bool = True
):
    """
    Train XGBoost classifier with cross-validation.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        scale_pos_weight: Weight for positive class (handles imbalance)
        verbose: Print progress messages

    Returns:
        Trained XGBoost model
    """
    if verbose:
        print("\n" + "="*80)
        print("TRAINING XGBOOST MODEL")
        print("="*80)

    # Calculate scale_pos_weight if not provided (handles class imbalance)
    if scale_pos_weight is None:
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1
        if verbose:
            print(f"\nClass imbalance ratio: {n_neg}:{n_pos} (neg:pos)")
            print(f"Setting scale_pos_weight = {scale_pos_weight:.2f}")

    # XGBoost hyperparameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': -1
    }

    if verbose:
        print("\nModel hyperparameters:")
        for key, value in params.items():
            print(f"  {key:20s} = {value}")

    # Train model
    if verbose:
        print("\nTraining XGBoost classifier...")

    model = xgb.XGBClassifier(**params)

    # Train with validation set
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=verbose
    )

    if verbose:
        print(f"\n✅ Training complete!")

    return model


def evaluate_model(model, X_test, y_test, feature_names, output_dir: Path):
    """
    Evaluate model performance and generate visualizations.

    Args:
        model: Trained XGBoost model
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        output_dir: Directory to save plots
    """
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Classification metrics
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nCONFUSION MATRIX:")
    print(f"                 Predicted")
    print(f"                 Legit  Phish")
    print(f"Actual Legit     {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"       Phish     {cm[1,0]:5d}  {cm[1,1]:5d}")

    # Additional metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)

    print(f"\nKEY METRICS:")
    print(f"  Precision:    {precision:.4f} (of predicted phishing, how many are correct)")
    print(f"  Recall:       {recall:.4f} (of actual phishing, how many are detected)")
    print(f"  F1 Score:     {f1:.4f} (harmonic mean of precision and recall)")
    print(f"  ROC-AUC:      {roc_auc:.4f} (overall discriminative ability)")
    print(f"  Avg Precision: {avg_precision:.4f} (precision-recall AUC)")

    # Visualizations
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
    print(f"\n✅ Saved confusion matrix to: {output_dir / 'confusion_matrix.png'}")
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve.png', dpi=150)
    print(f"✅ Saved ROC curve to: {output_dir / 'roc_curve.png'}")
    plt.close()

    # 3. Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall_vals, precision_vals, color='blue', lw=2,
            label=f'PR curve (AP = {avg_precision:.4f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall_curve.png', dpi=150)
    print(f"✅ Saved precision-recall curve to: {output_dir / 'precision_recall_curve.png'}")
    plt.close()

    # 4. Feature Importance
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(importance)), importance[indices], color='steelblue', alpha=0.8)
    ax.set_yticks(range(len(importance)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
    ax.set_xlabel('Importance (Gain)', fontsize=12)
    ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=150)
    print(f"✅ Saved feature importance to: {output_dir / 'feature_importance.png'}")
    plt.close()

    print("\n" + "="*80)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'confusion_matrix': cm.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost phishing detection model')
    parser.add_argument('--live-certs', type=Path,
                       default=Path('sources/raw/certs_live.jsonl'),
                       help='Path to live certificates JSONL file')
    parser.add_argument('--hist-certs', type=Path,
                       default=Path('sources/raw/certs_phishing_historical.jsonl'),
                       help='Path to historical phishing certificates JSONL file')
    parser.add_argument('--feature-stats', type=Path,
                       default=Path('sources/processed/comprehensive_feature_stats.csv'),
                       help='Path to comprehensive feature statistics CSV')
    parser.add_argument('--top-n', type=int, default=10,
                       help='Number of top features to use (by effect size)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data to use for testing')
    parser.add_argument('--output-dir', type=Path,
                       default=Path('models'),
                       help='Directory to save trained model and plots')

    args = parser.parse_args()

    print("="*80)
    print("XGBOOST PHISHING DETECTION MODEL TRAINING")
    print("="*80)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data and extract features
    df_combined, fe = load_data(args.live_certs, args.hist_certs)

    # Get top N features by effect size
    top_features = get_top_features(args.feature_stats, args.top_n)

    # Prepare training data
    print("\n" + "="*80)
    print("PREPARING TRAINING DATA")
    print("="*80)

    # Filter to top features
    X = df_combined[top_features].fillna(0)  # Fill NaN with 0
    y = df_combined['y']

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Label distribution:")
    print(f"  Legitimate (y=0): {(y == 0).sum():,}")
    print(f"  Phishing (y=1):   {(y == 1).sum():,}")

    # Train-test split (stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=42,
        stratify=y
    )

    print(f"\nTrain set: {len(X_train):,} samples")
    print(f"Test set:  {len(X_test):,} samples")

    # Train model
    model = train_model(X_train, y_train, X_test, y_test)

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test, top_features, args.output_dir)

    # Save model
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / 'xgboost_phishing_detector.joblib'
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved to: {model_path}")

    # Save feature list
    feature_list_path = args.output_dir / 'feature_list.json'
    with open(feature_list_path, 'w') as f:
        json.dump(top_features, f, indent=2)
    print(f"✅ Feature list saved to: {feature_list_path}")

    # Save metrics
    metrics_path = args.output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved to: {metrics_path}")

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModel artifacts saved to: {args.output_dir.absolute()}")
    print(f"  - xgboost_phishing_detector.joblib  (trained model)")
    print(f"  - feature_list.json                 (features used)")
    print(f"  - metrics.json                      (performance metrics)")
    print(f"  - confusion_matrix.png              (visualization)")
    print(f"  - roc_curve.png                     (visualization)")
    print(f"  - precision_recall_curve.png        (visualization)")
    print(f"  - feature_importance.png            (visualization)")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
