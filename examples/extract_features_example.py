#!/usr/bin/env python3
"""
Example: Extract Features from SSL Certificate Data

This script demonstrates how to use the FeatureEngineer class to extract
comprehensive features from SSL certificate data for phishing detection.

Usage:
    python examples/extract_features_example.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.features.feature_engineering import FeatureEngineer


def main():
    """Extract features from live and historical certificate data."""

    print("="*80)
    print("SSL Certificate Feature Extraction Example")
    print("="*80)

    # Initialize feature engineer (basic features only, no API calls)
    print("\n1. Initializing FeatureEngineer...")
    fe = FeatureEngineer(
        use_whois=False,      # Disable WHOIS (slow, rate-limited)
        use_abuseipdb=False,  # Disable AbuseIPDB (requires API key)
        use_greynoise=False   # Disable GreyNoise (requires API key)
    )

    # =========================================================================
    # Extract features from LIVE data
    # =========================================================================

    live_certs_path = Path("sources/raw/certs_live.jsonl")

    if live_certs_path.exists():
        print(f"\n2. Loading live certificates from {live_certs_path}...")
        df_live = pd.read_json(live_certs_path, lines=True)
        print(f"   Loaded {len(df_live):,} certificates")

        print("\n3. Extracting features from live data...")
        df_live_features = fe.extract_features(df_live, explode_domains=True, verbose=True)

        # Show sample
        print("\n   Sample features:")
        feature_cols = fe.get_feature_list()
        print(df_live_features[['domain'] + feature_cols[:10]].head().to_string())

        # Save to parquet
        output_path = Path("sources/processed/features_live.parquet")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_live_features.to_parquet(output_path, index=False)
        print(f"\n   ✅ Saved to {output_path}")
        print(f"      {len(df_live_features):,} rows × {len(feature_cols)} features")
    else:
        print(f"\n⚠️  Live cert file not found: {live_certs_path}")
        print("   Run ingestion first:")
        print("   python src/data/ingest_certificates_labels.py live-certs --duration 300")

    # =========================================================================
    # Extract features from HISTORICAL PHISHING data
    # =========================================================================

    hist_certs_path = Path("sources/raw/certs_phishing_historical.jsonl")

    if hist_certs_path.exists():
        print(f"\n4. Loading historical phishing certificates from {hist_certs_path}...")
        df_hist = pd.read_json(hist_certs_path, lines=True)
        print(f"   Loaded {len(df_hist):,} certificates")

        print("\n5. Extracting features from historical data...")
        df_hist_features = fe.extract_features(df_hist, explode_domains=True, verbose=True)

        # Show sample
        print("\n   Sample features:")
        print(df_hist_features[['domain'] + feature_cols[:10]].head().to_string())

        # Save to parquet
        output_path = Path("sources/processed/features_hist.parquet")
        df_hist_features.to_parquet(output_path, index=False)
        print(f"\n   ✅ Saved to {output_path}")
        print(f"      {len(df_hist_features):,} rows × {len(feature_cols)} features")
    else:
        print(f"\n⚠️  Historical cert file not found: {hist_certs_path}")
        print("   Run ingestion first:")
        print("   python src/data/ingest_certificates_labels.py historical-phishing-certs \\")
        print("       --labels sources/raw/phishing_labels.jsonl \\")
        print("       --output sources/raw/certs_phishing_historical.jsonl")

    # =========================================================================
    # Feature Summary
    # =========================================================================

    print("\n" + "="*80)
    print("FEATURE SUMMARY")
    print("="*80)

    all_features = fe.get_feature_list()

    print(f"\nTotal features extracted: {len(all_features)}")
    print("\nFeature categories:")
    print("  - Certificate-level: 7 features")
    print("    (cert_age_days, validity_days, is_wildcard, san_count,")
    print("     is_letsencrypt, has_subject_org, is_self_signed)")
    print("\n  - Domain-level: 15 features")
    print("    (domain_length, entropy, subdomain_count, hyphen_count,")
    print("     digit_count, digit_ratio, vowel_consonant_ratio,")
    print("     consecutive_consonants, has_at_symbol, has_ip_address,")
    print("     tld_risk, brand_distance, is_brand_lookalike,")
    print("     keyword_count, has_keyword)")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Run EDA notebook to analyze feature distributions:")
    print("   jupyter notebook notebooks/eda_sandbox.ipynb")
    print("\n2. Train ML models using extracted features:")
    print("   python src/models/train_model.py")
    print("\n3. For advanced features (WHOIS, AbuseIPDB, GreyNoise):")
    print("   - Get API keys (see src/features/README.md)")
    print("   - Re-run with use_whois=True, use_abuseipdb=True, etc.")


if __name__ == "__main__":
    main()
