"""
Feature Engineering Notebook
=============================
Run cells top to bottom. Each section builds on the previous.
Designed to work with:
    certs_ct.jsonl                     — live CT log sample (mostly y=0)
    certs_phishtank_historical.jsonl   — crt.sh phishing certs (y=1)
    phishtank_labels.jsonl             — domain-level labels

Install:
    pip install pandas numpy matplotlib seaborn python-Levenshtein tldextract
"""

# ── 0. Imports ─────────────────────────────────────────────────────────────────

import json
import math
import warnings
from collections import Counter
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Levenshtein import distance as levenshtein
import tldextract

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD AND COMBINE DATASETS
# ═══════════════════════════════════════════════════════════════════════════════

# Live CT log sample
df_live = pd.read_json("certs_fallback.jsonl", lines=True)
df_live["data_source"] = "ct_live"

# Historical phishing certs (already labeled y=1)
df_hist = pd.read_json("certs_phishtank_historical.jsonl", lines=True)
df_hist["data_source"] = "crtsh_historical"

# Labels for the live data
labels = pd.read_json("phishtank_labels.jsonl", lines=True)

print(f"Live certs:       {len(df_live):,}")
print(f"Historical certs: {len(df_hist):,}")
print(f"PhishTank labels: {len(labels):,}")

# ── Explode domains (one row per domain) ──────────────────────────────────────

def explode_domains(df):
    d = (df.explode("domains")
           .rename(columns={"domains": "domain"})
           .dropna(subset=["domain"]))
    d["domain"] = d["domain"].str.lower().str.strip().str.lstrip("*.")
    d = d[d["domain"].str.len() > 0]
    return d.reset_index(drop=True)

df_live_d = explode_domains(df_live)
df_hist_d = explode_domains(df_hist)

# Label live data via PhishTank join
df_live_d = df_live_d.merge(
    labels[["domain", "y", "label_source"]],
    on="domain", how="left"
)
df_live_d["y"] = df_live_d["y"].fillna(0).astype(int)

# Combine — historical is pre-labeled
df = pd.concat([df_live_d, df_hist_d], ignore_index=True)
df["y"] = df["y"].fillna(1).astype(int)  # hist records are all phishing

print(f"\nCombined domain rows: {len(df):,}")
print(df["y"].value_counts().rename({0: "legitimate/unknown", 1: "phishing"}))
print(f"\nPhishing rate: {df['y'].mean():.4%}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. RAW DATA INSPECTION
# Do this before engineering anything.
# ═══════════════════════════════════════════════════════════════════════════════

print("\n── Column dtypes ────────────────────────────────────────")
print(df.dtypes)

print("\n── Null counts ──────────────────────────────────────────")
print(df.isnull().sum())

print("\n── Sample phishing rows ─────────────────────────────────")
print(df[df["y"] == 1][["domain", "issuer", "not_before", "not_after"]].head(5).to_string())

print("\n── Sample legitimate rows ───────────────────────────────")
print(df[df["y"] == 0][["domain", "issuer", "not_before", "not_after"]].head(5).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# Add one feature at a time. Inspect distribution before moving on.
# ═══════════════════════════════════════════════════════════════════════════════

# ── 3a. Domain entropy ────────────────────────────────────────────────────────
# Measures randomness of characters. High entropy = random-looking = suspicious.
# Strip dots — they're structural, not informative.

def domain_entropy(domain: str) -> float:
    s = domain.replace(".", "")
    if not s:
        return 0.0
    counts = Counter(s)
    total = len(s)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())

df["entropy"] = df["domain"].apply(domain_entropy)

# Inspect
print("\n── Entropy by label ─────────────────────────────────────")
print(df.groupby("y")["entropy"].describe().round(3))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for label, ax, color in zip([0, 1], axes, ["steelblue", "tomato"]):
    df[df["y"] == label]["entropy"].hist(bins=40, ax=ax, color=color, alpha=0.8)
    ax.set_title(f"Entropy — {'Phishing' if label else 'Legitimate'}")
    ax.set_xlabel("Entropy")
plt.tight_layout()
plt.savefig("feat_entropy.png", dpi=100)
plt.show()
# ⚠️  Expected finding: entropy alone is noisy. Many legit CDN/tracking domains
#     are high entropy. Use as one input to the model, not a standalone filter.

# ── 3b. Subdomain count ───────────────────────────────────────────────────────
# Phishing often uses deep subdomain structures to bury the malicious part:
# e.g. paypal.secure-login.malicious-site.com

def subdomain_count(domain: str) -> int:
    ext = tldextract.extract(domain)
    if not ext.subdomain:
        return 0
    return len(ext.subdomain.split("."))

df["subdomain_count"] = df["domain"].apply(subdomain_count)

print("\n── Subdomain count by label ─────────────────────────────")
print(df.groupby("y")["subdomain_count"].value_counts().unstack(fill_value=0).head(6))

# ── 3c. Domain length ─────────────────────────────────────────────────────────
# Phishing domains trend longer — stuffed with brand keywords + random strings.

df["domain_length"] = df["domain"].str.len()

print("\n── Domain length by label ───────────────────────────────")
print(df.groupby("y")["domain_length"].describe().round(1))

# ── 3d. TLD risk ──────────────────────────────────────────────────────────────
# Tiered risk: some TLDs are overwhelmingly abused, some are trust signals.

HIGH_RISK_TLDS  = {"top", "xyz", "buzz", "click", "live", "online", "site",
                   "club", "work", "shop", "icu", "vip", "fun", "today"}
LOW_RISK_TLDS   = {"com", "org", "net", "edu", "gov", "co", "io", "dev"}
TRUST_TLDS      = {"gov", "edu", "mil"}

def tld_risk(domain: str) -> int:
    """2 = high risk, 1 = neutral, 0 = low risk / trust signal"""
    ext = tldextract.extract(domain)
    tld = ext.suffix.lower().split(".")[-1] if ext.suffix else ""
    if tld in TRUST_TLDS:
        return 0
    if tld in HIGH_RISK_TLDS:
        return 2
    if tld in LOW_RISK_TLDS:
        return 1
    return 1  # unknown TLD → neutral

df["tld"] = df["domain"].apply(lambda d: tldextract.extract(d).suffix.lower())
df["tld_risk"] = df["domain"].apply(tld_risk)

print("\n── TLD risk distribution by label ───────────────────────")
print(df.groupby("y")["tld_risk"].value_counts(normalize=True).unstack().round(3))

print("\n── Top TLDs in phishing certs ───────────────────────────")
print(df[df["y"] == 1]["tld"].value_counts().head(15))

# ── 3e. Issuer features ───────────────────────────────────────────────────────

def issuer_org(row) -> str | None:
    if isinstance(row, dict):
        return row.get("O")
    return None

df["issuer_org"]      = df["issuer"].apply(issuer_org)
df["is_letsencrypt"]  = df["issuer_org"].str.contains("Let's Encrypt", na=False).astype(int)
df["has_org"]         = df["issuer_org"].notna().astype(int)  # OV/EV cert signal

print("\n── Let's Encrypt rate by label ──────────────────────────")
print(df.groupby("y")["is_letsencrypt"].mean().round(4))

print("\n── Has organisation (OV/EV) by label ───────────────────")
print(df.groupby("y")["has_org"].mean().round(4))

# ── 3f. Validity duration ─────────────────────────────────────────────────────
# LE max is 90 days. Legit long-running services often get 1yr certs.
# Very short validity on a suspicious domain = strong signal.

df["validity_days"] = ((df["not_after"] - df["not_before"]) / 86400).round(1)
df["validity_days"] = df["validity_days"].clip(lower=0, upper=3650)  # cap outliers

print("\n── Validity duration by label (days) ────────────────────")
print(df.groupby("y")["validity_days"].describe().round(1))

# ── 3g. Brand distance ────────────────────────────────────────────────────────
# Minimum Levenshtein distance from the registrable domain to any brand name.
# Low distance = looks like a known brand = suspicious.
# Compute on registrable domain only (not subdomains).

BRANDS = [
    "paypal", "amazon", "apple", "microsoft", "google", "facebook",
    "netflix", "instagram", "linkedin", "twitter", "dropbox", "gmail",
    "outlook", "office365", "wellsfargo", "chase", "bankofamerica",
    "coinbase", "binance", "metamask",
]

def min_brand_distance(domain: str) -> int:
    ext = tldextract.extract(domain)
    base = ext.domain.lower() if ext.domain else domain
    return min(levenshtein(base, brand) for brand in BRANDS)

def closest_brand(domain: str) -> str:
    ext = tldextract.extract(domain)
    base = ext.domain.lower() if ext.domain else domain
    return min(BRANDS, key=lambda b: levenshtein(base, b))

# NOTE: This is the slowest feature (~1-2 min on 100k rows).
# Run once, cache to disk.
print("\nComputing brand distances (slow — ~1-2 min) …")
df["brand_distance"]       = df["domain"].apply(min_brand_distance)
df["closest_brand"]        = df["domain"].apply(closest_brand)
df["is_brand_lookalike"]   = (df["brand_distance"] <= 3).astype(int)

print("\n── Brand distance by label ──────────────────────────────")
print(df.groupby("y")["brand_distance"].describe().round(2))

print("\n── Brand lookalike rate by label ────────────────────────")
print(df.groupby("y")["is_brand_lookalike"].mean().round(4))

print("\n── Most targeted brands in phishing ─────────────────────")
print(df[df["y"] == 1]["closest_brand"].value_counts().head(10))

# ── 3h. Keyword presence ──────────────────────────────────────────────────────
# Phishing domains often contain trust-inducing words.

PHISHING_KEYWORDS = [
    "secure", "login", "verify", "account", "update", "banking",
    "signin", "wallet", "confirm", "support", "help", "service",
    "alert", "payment", "invoice",
]

def keyword_count(domain: str) -> int:
    return sum(kw in domain for kw in PHISHING_KEYWORDS)

df["keyword_count"] = df["domain"].apply(keyword_count)
df["has_keyword"]   = (df["keyword_count"] > 0).astype(int)

print("\n── Keyword presence by label ────────────────────────────")
print(df.groupby("y")["has_keyword"].mean().round(4))

# ═══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE SUMMARY AND CORRELATION
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "entropy", "subdomain_count", "domain_length", "tld_risk",
    "is_letsencrypt", "has_org", "validity_days", "brand_distance",
    "is_brand_lookalike", "keyword_count",
]

print("\n── Feature means by label ───────────────────────────────")
print(df.groupby("y")[FEATURE_COLS].mean().round(3).T.rename(columns={0: "legit", 1: "phishing"}))

# Correlation of each feature with label
print("\n── Correlation with y (label) ───────────────────────────")
corr = df[FEATURE_COLS + ["y"]].corr()["y"].drop("y").sort_values(key=abs, ascending=False)
print(corr.round(3))

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    df[FEATURE_COLS + ["y"]].corr(),
    annot=True, fmt=".2f", cmap="RdBu_r", center=0,
    square=True, linewidths=0.5,
)
plt.title("Feature correlation matrix")
plt.tight_layout()
plt.savefig("feat_correlation.png", dpi=100)
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# 5. SAVE FEATURE MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

# Save full feature matrix — use for model training in next notebook
df_features = df[["domain", "fingerprint", "data_source", "y"] + FEATURE_COLS].copy()
df_features.to_parquet("features.parquet", index=False)
print(f"\nSaved feature matrix: features.parquet ({len(df_features):,} rows)")

# Save only live data for time series work
df_ts = df[df["data_source"] == "ct_live"][["domain", "timestamp", "y"] + FEATURE_COLS].copy()
df_ts.to_parquet("features_timeseries.parquet", index=False)
print(f"Saved time series slice: features_timeseries.parquet ({len(df_ts):,} rows)")

print("\n── Next steps ───────────────────────────────────────────")
print("1. Review feat_entropy.png and feat_correlation.png")
print("2. Drop features with |correlation| < 0.02 — they add noise not signal")
print("3. Move to model_training.py with features.parquet")
print("4. Use features_timeseries.parquet for campaign/anomaly work")
