# Feature Engineering for SSL Anomaly Detection

Comprehensive feature extraction from SSL certificates and domains for phishing detection.

## Features Overview

### Certificate-Level Features (9 features)

| Feature | Type | Description | Typical Values |
|---------|------|-------------|----------------|
| `cert_age_days` | float | Days since certificate was issued | 0-365+ |
| `validity_days` | float | Certificate validity period (days) | Let's Encrypt: ~90, Traditional: 365-397 |
| `is_wildcard` | binary | Uses wildcard domain (*.example.com) | 0 or 1 |
| `san_count` | int | Number of Subject Alternative Names | Phishing: 1-2, Legit multi-domain: 10+ |
| `is_letsencrypt` | binary | Issued by Let's Encrypt | 0 or 1 |
| `has_subject_org` | binary | Subject has organization field (OV/EV cert) | 0 or 1 |
| `is_self_signed` | binary | Self-signed certificate | 0 or 1 |
| `issuer_org` | string | Issuer organization name | "Let's Encrypt", "DigiCert", etc. |
| `subject_org` | string | Subject organization (if present) | Company name or None |
| `subject_ou` | string | Subject organizational unit | Department/division or None |
| `subject_country` | string | Subject country code | "US", "CN", etc. |

### Domain-Level Features (15+ features)

| Feature | Type | Description | Phishing Indicator |
|---------|------|-------------|-------------------|
| `domain_length` | int | Total domain length | Phishing often longer (>30 chars) |
| `entropy` | float | Shannon entropy (0-5) | High entropy = random-looking |
| `subdomain_count` | int | Number of subdomain levels | Deep subdomains suspicious (>3) |
| `hyphen_count` | int | Number of hyphens | Phishing often uses hyphens (>2) |
| `digit_count` | int | Number of digits | Excessive digits suspicious (>5) |
| `digit_ratio` | float | Ratio of digits to total chars | High ratio suspicious (>0.3) |
| `vowel_consonant_ratio` | float | Vowels/consonants ratio | Unusual ratios (very high/low) |
| `consecutive_consonants` | int | Max consecutive consonants | Legitimate <4, suspicious >5 |
| `has_at_symbol` | binary | Contains @ symbol | URL confusion trick |
| `has_ip_address` | binary | Contains IP address pattern | IP instead of domain name |
| `tld` | string | Top-level domain | ".com", ".xyz", ".top", etc. |
| `tld_risk` | int | TLD risk score (0-2) | 0=trusted, 1=neutral, 2=high-risk |
| `brand_distance` | int | Min Levenshtein distance to brand | ≤3 = potential typosquatting |
| `closest_brand` | string | Closest brand name | "paypal", "amazon", etc. |
| `is_brand_lookalike` | binary | Distance ≤3 from known brand | 0 or 1 |
| `keyword_count` | int | Count of phishing keywords | "secure", "login", "verify", etc. |
| `has_keyword` | binary | Has any phishing keyword | 0 or 1 |

### Optional External Features (Requires API Keys)

| Feature | Type | Description | API Required | Rate Limits |
|---------|------|-------------|--------------|-------------|
| `domain_age_days` | float | Domain age from WHOIS | python-whois | ~100/day |
| `abuseipdb_score` | int | Abuse confidence score (0-100) | AbuseIPDB | 1000/day (free) |
| `greynoise_classification` | string | Scanner classification | GreyNoise | 50/day (free) |

## Installation

```bash
pip install pandas numpy python-Levenshtein tldextract

# Optional dependencies
pip install python-whois requests
```

## Usage

### Basic Usage

```python
from src.features.feature_engineering import FeatureEngineer
import pandas as pd

# Load certificate data
df = pd.read_json("sources/raw/certs_live.jsonl", lines=True)

# Initialize feature engineer
fe = FeatureEngineer()

# Extract features
df_features = fe.extract_features(df)

# Save to parquet
df_features.to_parquet("sources/processed/features.parquet", index=False)
```

### Command-Line Usage

```bash
# Extract features from JSONL file
python src/features/feature_engineering.py \
    sources/raw/certs_live.jsonl \
    sources/processed/features_live.parquet
```

### With Optional External Features

```python
fe = FeatureEngineer(
    use_whois=True,              # Enable WHOIS domain age lookup
    use_abuseipdb=True,          # Enable AbuseIPDB abuse score
    use_greynoise=True,          # Enable GreyNoise scanner detection
    abuseipdb_key="YOUR_KEY",    # Get free key: https://www.abuseipdb.com/api
    greynoise_key="YOUR_KEY"     # Get free key: https://www.greynoise.io/
)

df_features = fe.extract_features(df)
```

### Get Feature List for Model Training

```python
# Get list of all numeric feature columns
features = fe.get_feature_list()

# Use for model training
X = df_features[features]
y = df_features['y']
```

## Feature Engineering Pipeline

### 1. Extract Features from Live Data

```bash
# Collect live certs
python src/data/ingest_certificates_labels.py live-certs \
    --duration 900 \
    --output sources/raw/certs_live.jsonl

# Extract features
python src/features/feature_engineering.py \
    sources/raw/certs_live.jsonl \
    sources/processed/features_live.parquet
```

### 2. Extract Features from Historical Phishing Data

```bash
# Collect historical phishing certs
python src/data/ingest_certificates_labels.py historical-phishing-certs \
    --labels sources/raw/phishing_labels.jsonl \
    --output sources/raw/certs_phishing_historical.jsonl

# Extract features
python src/features/feature_engineering.py \
    sources/raw/certs_phishing_historical.jsonl \
    sources/processed/features_hist.parquet
```

### 3. Merge Live and Historical for Training

```python
import pandas as pd

# Load features
df_live = pd.read_parquet("sources/processed/features_live.parquet")
df_hist = pd.read_parquet("sources/processed/features_hist.parquet")

# Merge
df_train = pd.concat([df_live, df_hist], ignore_index=True)

# Save
df_train.to_parquet("sources/processed/features_train.parquet", index=False)
```

## Feature Importance (Expected Insights)

Based on phishing research, the most discriminative features are typically:

### High Importance
- `brand_distance` - Typosquatting detection
- `is_brand_lookalike` - Lookalike domains
- `keyword_count` - Phishing keywords ("secure", "login", etc.)
- `validity_days` - Short-lived certs
- `entropy` - Random-looking domains
- `tld_risk` - High-risk TLDs (.xyz, .top)

### Medium Importance
- `subdomain_count` - Deep subdomain abuse
- `hyphen_count` - Excessive hyphens
- `digit_ratio` - High digit content
- `is_letsencrypt` - Free cert indicator
- `san_count` - Single vs multi-domain
- `has_subject_org` - DV vs OV/EV cert

### Low Importance (Noisy)
- `domain_length` - Many legitimate CDN domains are long
- `vowel_consonant_ratio` - Language-dependent
- `cert_age_days` - Both old and new certs can be malicious

## API Key Setup

### AbuseIPDB (Free Tier: 1,000 requests/day)

1. Sign up: https://www.abuseipdb.com/register
2. Get API key: https://www.abuseipdb.com/account/api
3. Store in environment:
   ```bash
   export ABUSEIPDB_KEY="your_key_here"
   ```

### GreyNoise (Free Tier: 50 requests/day)

1. Sign up: https://viz.greynoise.io/signup
2. Get API key: https://viz.greynoise.io/account
3. Store in environment:
   ```bash
   export GREYNOISE_KEY="your_key_here"
   ```

### WHOIS (No API Key Required)

WHOIS lookups use public WHOIS servers, but may be rate-limited by registrars:
- Free tier: ~100-1000 queries/day depending on registrar
- Consider caching results for repeated analysis

## Performance Notes

- **Basic features**: ~100-1000 rows/sec
- **Brand distance**: ~10-50 rows/sec (Levenshtein computation)
- **WHOIS lookups**: ~1-5 rows/sec (network latency, rate limits)
- **AbuseIPDB/GreyNoise**: ~1 row/sec (API rate limits)

**Recommendation**: Extract basic features first, then add external features only for final dataset preparation.

## Example Output

```python
df_features.head()
```

| domain | entropy | subdomain_count | brand_distance | validity_days | is_letsencrypt | san_count | y |
|--------|---------|----------------|----------------|---------------|----------------|-----------|---|
| paypa1-secure.com | 3.89 | 0 | 1 | 90 | 1 | 1 | 1 |
| amazon.com | 2.52 | 0 | 0 | 397 | 0 | 12 | 0 |
| secure-login-apple.xyz | 4.12 | 0 | 4 | 90 | 1 | 1 | 1 |
| google.com | 2.25 | 0 | 0 | 397 | 0 | 8 | 0 |

## Next Steps

1. **EDA**: Use `notebooks/eda_sandbox.ipynb` to analyze feature distributions
2. **Model Training**: Train classifiers using extracted features
3. **Feature Selection**: Use statistical tests to identify most discriminative features
4. **Real-Time Detection**: Integrate feature extraction into live ingestion pipeline
