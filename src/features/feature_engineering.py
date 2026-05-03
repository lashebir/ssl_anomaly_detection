"""
Feature Engineering for SSL Anomaly Detection
==============================================

Comprehensive feature extraction from SSL certificates and domains including:
- SSL certificate-level features (age, validity, issuer, wildcards, SANs, org fields)
- Domain-level features (entropy, hyphens, digits, misspellings, TLD, Levenshtein)
- IP-level features (abuse status, scanner detection) - optional, requires API keys

Usage:
    from src.features.feature_engineering import FeatureEngineer

    fe = FeatureEngineer()
    df = pd.read_json("certs.jsonl", lines=True)
    df_features = fe.extract_features(df)

Install:
    pip install pandas numpy python-Levenshtein tldextract whois requests
"""

import json
import math
import warnings
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import time

import numpy as np
import pandas as pd
from Levenshtein import distance as levenshtein
import tldextract

warnings.filterwarnings("ignore")


class FeatureEngineer:
    """
    Comprehensive feature extraction for SSL certificate anomaly detection.

    Features extracted:
    - Certificate-level: age, validity, issuer, wildcards, SANs, org fields
    - Domain-level: entropy, hyphens, digits, misspellings, TLD, brand distance
    - IP-level (optional): abuse status, scanner detection
    """

    # Known high-risk TLDs (frequently abused)
    HIGH_RISK_TLDS = {
        "top", "xyz", "buzz", "click", "live", "online", "site",
        "club", "work", "shop", "icu", "vip", "fun", "today",
        "tk", "ml", "ga", "cf", "gq", "pw", "cc"
    }

    # Trusted/low-risk TLDs
    LOW_RISK_TLDS = {"com", "org", "net", "edu", "gov", "co", "io", "dev", "app"}
    TRUST_TLDS = {"gov", "edu", "mil"}

    # Brand names for typosquatting detection
    BRANDS = [
        "paypal", "amazon", "apple", "microsoft", "google", "facebook",
        "netflix", "instagram", "linkedin", "twitter", "dropbox", "gmail",
        "outlook", "office365", "wellsfargo", "chase", "bankofamerica",
        "coinbase", "binance", "metamask", "stripe", "shopify", "ebay",
        "alibaba", "tencent", "baidu", "yahoo", "zoom", "slack"
    ]

    # Phishing keywords
    PHISHING_KEYWORDS = [
        "secure", "login", "verify", "account", "update", "banking",
        "signin", "wallet", "confirm", "support", "help", "service",
        "alert", "payment", "invoice", "validate", "suspend", "unlock"
    ]

    def __init__(
        self,
        use_whois: bool = False,
        use_abuseipdb: bool = False,
        use_greynoise: bool = False,
        abuseipdb_key: Optional[str] = None,
        greynoise_key: Optional[str] = None
    ):
        """
        Initialize feature engineer.

        Args:
            use_whois: Enable WHOIS domain age lookup (slow, may be rate-limited)
            use_abuseipdb: Enable AbuseIPDB abuse score lookup (requires API key)
            use_greynoise: Enable GreyNoise scanner detection (requires API key)
            abuseipdb_key: AbuseIPDB API key (free tier: 1000 requests/day)
            greynoise_key: GreyNoise API key (free tier: 50 requests/day)
        """
        self.use_whois = use_whois
        self.use_abuseipdb = use_abuseipdb
        self.use_greynoise = use_greynoise
        self.abuseipdb_key = abuseipdb_key
        self.greynoise_key = greynoise_key

        # Cache for expensive lookups
        self._whois_cache = {}
        self._abuseipdb_cache = {}
        self._greynoise_cache = {}

    # =========================================================================
    # DOMAIN-LEVEL FEATURES
    # =========================================================================

    @staticmethod
    def domain_entropy(domain: str) -> float:
        """
        Calculate Shannon entropy of domain (excluding dots).
        Higher entropy = more random = suspicious.

        Args:
            domain: Domain name (e.g., "example.com")

        Returns:
            Entropy value (typically 0-5)
        """
        s = domain.replace(".", "")
        if not s:
            return 0.0
        counts = Counter(s)
        total = len(s)
        return -sum((c / total) * math.log2(c / total) for c in counts.values())

    @staticmethod
    def subdomain_count(domain: str) -> int:
        """
        Count number of subdomain levels.

        Examples:
            www.example.com → 1
            mail.secure.example.com → 2
            example.com → 0
        """
        ext = tldextract.extract(domain)
        if not ext.subdomain:
            return 0
        return len(ext.subdomain.split("."))

    @staticmethod
    def hyphen_count(domain: str) -> int:
        """Count hyphens in domain (often used in phishing)."""
        return domain.count("-")

    @staticmethod
    def digit_count(domain: str) -> int:
        """Count digits in domain."""
        return sum(c.isdigit() for c in domain)

    @staticmethod
    def digit_ratio(domain: str) -> float:
        """Ratio of digits to total characters (excluding dots)."""
        s = domain.replace(".", "")
        if not s:
            return 0.0
        return sum(c.isdigit() for c in s) / len(s)

    @staticmethod
    def vowel_consonant_ratio(domain: str) -> float:
        """
        Ratio of vowels to consonants (excluding dots and digits).
        Suspicious domains may have unusual ratios.
        """
        s = domain.replace(".", "").lower()
        letters = [c for c in s if c.isalpha()]
        if not letters:
            return 0.0
        vowels = sum(c in "aeiou" for c in letters)
        consonants = len(letters) - vowels
        return vowels / consonants if consonants > 0 else 0.0

    @staticmethod
    def consecutive_consonants(domain: str) -> int:
        """
        Max consecutive consonants.
        Legitimate domains rarely have >4 consecutive consonants.
        """
        s = domain.replace(".", "").lower()
        max_run = 0
        current_run = 0

        for c in s:
            if c.isalpha() and c not in "aeiou":
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0

        return max_run

    def tld_risk(self, domain: str) -> int:
        """
        TLD risk score.

        Returns:
            0 = trusted (gov/edu/mil)
            1 = low-risk (com/org/net)
            2 = high-risk (xyz/top/tk)
        """
        ext = tldextract.extract(domain)
        tld = ext.suffix.lower().split(".")[-1] if ext.suffix else ""

        if tld in self.TRUST_TLDS:
            return 0
        if tld in self.HIGH_RISK_TLDS:
            return 2
        if tld in self.LOW_RISK_TLDS:
            return 1
        return 1  # unknown → neutral

    def min_brand_distance(self, domain: str) -> int:
        """
        Minimum Levenshtein distance from registrable domain to any brand.
        Lower distance = potential typosquatting.

        Examples:
            "paypa1.com" → distance 1 from "paypal"
            "amazon-secure.com" → distance 0 from "amazon" (subdomain)
        """
        ext = tldextract.extract(domain)
        base = ext.domain.lower() if ext.domain else domain
        if not base:
            return 999
        return min(levenshtein(base, brand) for brand in self.BRANDS)

    def closest_brand(self, domain: str) -> str:
        """Return the brand name with minimum Levenshtein distance."""
        ext = tldextract.extract(domain)
        base = ext.domain.lower() if ext.domain else domain
        if not base:
            return "none"
        return min(self.BRANDS, key=lambda b: levenshtein(base, b))

    def keyword_count(self, domain: str) -> int:
        """Count phishing-related keywords in domain."""
        domain_lower = domain.lower()
        return sum(kw in domain_lower for kw in self.PHISHING_KEYWORDS)

    @staticmethod
    def has_at_symbol(domain: str) -> int:
        """Check for @ symbol (URL confusion trick)."""
        return int("@" in domain)

    @staticmethod
    def has_ip_address(domain: str) -> int:
        """
        Check if domain contains IP address pattern.
        Phishing often uses IP addresses instead of domains.
        """
        import re
        ip_pattern = r'\d{1,3}[-\.]\d{1,3}[-\.]\d{1,3}[-\.]\d{1,3}'
        return int(bool(re.search(ip_pattern, domain)))

    # =========================================================================
    # CERTIFICATE-LEVEL FEATURES
    # =========================================================================

    @staticmethod
    def cert_age_days(not_before: int, reference_time: Optional[int] = None) -> float:
        """
        Certificate age in days from not_before to reference time.

        Args:
            not_before: Unix timestamp (not_before field)
            reference_time: Unix timestamp (default: current time)

        Returns:
            Age in days
        """
        if reference_time is None:
            reference_time = int(datetime.now(timezone.utc).timestamp())
        return max(0, (reference_time - not_before) / 86400)

    @staticmethod
    def validity_days(not_before: int, not_after: int) -> float:
        """
        Certificate validity period in days.
        Let's Encrypt: max 90 days
        Traditional CAs: often 365-397 days
        """
        return max(0, (not_after - not_before) / 86400)

    @staticmethod
    def is_wildcard(domains: List[str]) -> int:
        """Check if certificate uses wildcard (*.example.com)."""
        if not domains:
            return 0
        return int(any(d.startswith("*.") or d.startswith("*") for d in domains))

    @staticmethod
    def san_count(domains: List[str]) -> int:
        """
        Count Subject Alternative Names (SANs).
        Legitimate multi-domain certs may have many SANs.
        Phishing often has 1-2 SANs.
        """
        return len(domains) if domains else 0

    @staticmethod
    def extract_issuer_org(issuer: Dict[str, Any]) -> Optional[str]:
        """Extract issuer organization name."""
        if isinstance(issuer, dict):
            return issuer.get("O")
        return None

    @staticmethod
    def extract_subject_org(subject: Dict[str, Any]) -> Optional[str]:
        """Extract subject organization name (OV/EV certs only)."""
        if isinstance(subject, dict):
            return subject.get("O")
        return None

    @staticmethod
    def extract_subject_ou(subject: Dict[str, Any]) -> Optional[str]:
        """Extract subject organizational unit."""
        if isinstance(subject, dict):
            return subject.get("OU")
        return None

    @staticmethod
    def extract_subject_country(subject: Dict[str, Any]) -> Optional[str]:
        """Extract subject country code."""
        if isinstance(subject, dict):
            return subject.get("C")
        return None

    @staticmethod
    def is_letsencrypt(issuer_org: Optional[str]) -> int:
        """Check if issued by Let's Encrypt."""
        if not issuer_org:
            return 0
        return int("let's encrypt" in issuer_org.lower())

    @staticmethod
    def is_self_signed(subject: Dict[str, Any], issuer: Dict[str, Any]) -> int:
        """
        Check if certificate appears self-signed.
        Self-signed certs have identical subject and issuer.
        """
        if not isinstance(subject, dict) or not isinstance(issuer, dict):
            return 0
        return int(subject.get("CN") == issuer.get("CN") and
                  subject.get("O") == issuer.get("O"))

    @staticmethod
    def has_subject_org(subject: Dict[str, Any]) -> int:
        """
        Check if subject has organization field.
        OV/EV certs have this; DV certs do not.
        """
        if isinstance(subject, dict):
            return int(subject.get("O") is not None)
        return 0

    # =========================================================================
    # EXTERNAL DATA FEATURES (OPTIONAL)
    # =========================================================================

    def domain_age_whois(self, domain: str) -> Optional[float]:
        """
        Get domain age in days from WHOIS.

        WARNING: Slow and may be rate-limited.
        Free tier: ~100 queries/day depending on registrar.

        Returns:
            Age in days, or None if lookup fails
        """
        if not self.use_whois:
            return None

        if domain in self._whois_cache:
            return self._whois_cache[domain]

        try:
            import whois
            w = whois.whois(domain)
            if w.creation_date:
                creation = w.creation_date
                if isinstance(creation, list):
                    creation = creation[0]
                age_days = (datetime.now(timezone.utc) - creation).days
                self._whois_cache[domain] = age_days
                return age_days
        except Exception as e:
            print(f"WHOIS lookup failed for {domain}: {e}")
            self._whois_cache[domain] = None

        return None

    def abuseipdb_score(self, ip: str) -> Optional[int]:
        """
        Get AbuseIPDB abuse confidence score (0-100).

        Requires: AbuseIPDB API key
        Free tier: 1000 requests/day

        Returns:
            Abuse score (0-100), or None if lookup fails
        """
        if not self.use_abuseipdb or not self.abuseipdb_key:
            return None

        if ip in self._abuseipdb_cache:
            return self._abuseipdb_cache[ip]

        try:
            import requests
            url = "https://api.abuseipdb.com/api/v2/check"
            headers = {
                "Key": self.abuseipdb_key,
                "Accept": "application/json"
            }
            params = {"ipAddress": ip, "maxAgeInDays": 90}

            response = requests.get(url, headers=headers, params=params, timeout=5)
            response.raise_for_status()

            data = response.json()
            score = data.get("data", {}).get("abuseConfidenceScore", 0)
            self._abuseipdb_cache[ip] = score

            # Rate limit: 1 req/sec on free tier
            time.sleep(1)
            return score
        except Exception as e:
            print(f"AbuseIPDB lookup failed for {ip}: {e}")
            self._abuseipdb_cache[ip] = None

        return None

    def greynoise_classification(self, ip: str) -> Optional[str]:
        """
        Get GreyNoise classification (benign/malicious/unknown).

        Requires: GreyNoise API key
        Free tier: 50 requests/day

        Returns:
            "benign", "malicious", "unknown", or None if lookup fails
        """
        if not self.use_greynoise or not self.greynoise_key:
            return None

        if ip in self._greynoise_cache:
            return self._greynoise_cache[ip]

        try:
            import requests
            url = f"https://api.greynoise.io/v3/community/{ip}"
            headers = {"key": self.greynoise_key}

            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()

            data = response.json()
            classification = data.get("classification", "unknown")
            self._greynoise_cache[ip] = classification

            # Rate limit: 1 req/sec recommended
            time.sleep(1)
            return classification
        except Exception as e:
            print(f"GreyNoise lookup failed for {ip}: {e}")
            self._greynoise_cache[ip] = None

        return None

    # =========================================================================
    # MAIN FEATURE EXTRACTION
    # =========================================================================

    def extract_domain_features(self, row: pd.Series) -> Dict[str, Any]:
        """Extract all domain-level features from a certificate row."""
        domain = row.get("domain", "")

        features = {
            # Basic domain features
            "domain_length": len(domain),
            "entropy": self.domain_entropy(domain),
            "subdomain_count": self.subdomain_count(domain),
            "hyphen_count": self.hyphen_count(domain),
            "digit_count": self.digit_count(domain),
            "digit_ratio": self.digit_ratio(domain),

            # Advanced domain features
            "vowel_consonant_ratio": self.vowel_consonant_ratio(domain),
            "consecutive_consonants": self.consecutive_consonants(domain),
            "has_at_symbol": self.has_at_symbol(domain),
            "has_ip_address": self.has_ip_address(domain),

            # TLD features
            "tld": tldextract.extract(domain).suffix.lower(),
            "tld_risk": self.tld_risk(domain),

            # Brand/typosquatting features
            "brand_distance": self.min_brand_distance(domain),
            "closest_brand": self.closest_brand(domain),
            "is_brand_lookalike": int(self.min_brand_distance(domain) <= 3),

            # Keyword features
            "keyword_count": self.keyword_count(domain),
            "has_keyword": int(self.keyword_count(domain) > 0),
        }

        # Optional: WHOIS domain age
        if self.use_whois:
            features["domain_age_days"] = self.domain_age_whois(domain)

        return features

    def extract_cert_features(self, row: pd.Series) -> Dict[str, Any]:
        """Extract all certificate-level features from a certificate row."""
        features = {
            # Certificate age and validity
            "cert_age_days": self.cert_age_days(row.get("not_before", 0)),
            "validity_days": self.validity_days(
                row.get("not_before", 0),
                row.get("not_after", 0)
            ),

            # Wildcard and SAN features
            "is_wildcard": self.is_wildcard(row.get("domains", [])),
            "san_count": self.san_count(row.get("domains", [])),

            # Issuer features
            "issuer_org": self.extract_issuer_org(row.get("issuer", {})),
            "is_letsencrypt": self.is_letsencrypt(
                self.extract_issuer_org(row.get("issuer", {}))
            ),

            # Subject features (OV/EV signals)
            "subject_org": self.extract_subject_org(row.get("subject", {})),
            "subject_ou": self.extract_subject_ou(row.get("subject", {})),
            "subject_country": self.extract_subject_country(row.get("subject", {})),
            "has_subject_org": self.has_subject_org(row.get("subject", {})),

            # Self-signed check
            "is_self_signed": self.is_self_signed(
                row.get("subject", {}),
                row.get("issuer", {})
            ),
        }

        return features

    def extract_features(
        self,
        df: pd.DataFrame,
        explode_domains: bool = True,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Extract all features from certificate dataframe.

        Args:
            df: Certificate dataframe with columns: domains, not_before, not_after,
                issuer, subject, etc.
            explode_domains: If True, explode multi-domain certs to one row per domain
            verbose: Print progress messages

        Returns:
            DataFrame with all features extracted
        """
        if verbose:
            print(f"Extracting features from {len(df):,} certificates...")

        df_out = df.copy()

        # Explode domains if requested
        if explode_domains:
            if verbose:
                print("  - Exploding domains (one row per domain)...")
            df_out = (df_out
                .explode("domains")
                .rename(columns={"domains": "domain"})
                .dropna(subset=["domain"])
                .reset_index(drop=True)
            )
            df_out["domain"] = (df_out["domain"]
                .str.lower()
                .str.strip()
                .str.lstrip("*.")
            )
            df_out = df_out[df_out["domain"].str.len() > 0]

        # Extract certificate features
        if verbose:
            print("  - Extracting certificate features...")
        cert_features = df_out.apply(self.extract_cert_features, axis=1, result_type="expand")
        df_out = pd.concat([df_out, cert_features], axis=1)

        # Extract domain features
        if verbose:
            print("  - Extracting domain features...")
        domain_features = df_out.apply(self.extract_domain_features, axis=1, result_type="expand")
        df_out = pd.concat([df_out, domain_features], axis=1)

        if verbose:
            print(f"✅ Feature extraction complete: {len(df_out):,} rows, "
                  f"{len(cert_features.columns) + len(domain_features.columns)} features")

        return df_out

    def get_feature_list(self) -> List[str]:
        """
        Get list of all feature column names.
        Useful for model training.
        """
        features = [
            # Domain features
            "domain_length", "entropy", "subdomain_count", "hyphen_count",
            "digit_count", "digit_ratio", "vowel_consonant_ratio",
            "consecutive_consonants", "has_at_symbol", "has_ip_address",
            "tld_risk", "brand_distance", "is_brand_lookalike",
            "keyword_count", "has_keyword",

            # Certificate features
            "cert_age_days", "validity_days", "is_wildcard", "san_count",
            "is_letsencrypt", "has_subject_org", "is_self_signed",
        ]

        # Add optional features
        if self.use_whois:
            features.append("domain_age_days")

        return features


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def extract_features_from_jsonl(
    input_path: str,
    output_path: str,
    use_whois: bool = False,
    use_abuseipdb: bool = False,
    use_greynoise: bool = False,
    abuseipdb_key: Optional[str] = None,
    greynoise_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Extract features from JSONL certificate file and save to parquet.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output parquet file
        use_whois: Enable WHOIS lookups
        use_abuseipdb: Enable AbuseIPDB lookups
        use_greynoise: Enable GreyNoise lookups
        abuseipdb_key: AbuseIPDB API key
        greynoise_key: GreyNoise API key

    Returns:
        DataFrame with extracted features
    """
    print(f"Loading certificates from {input_path}...")
    df = pd.read_json(input_path, lines=True)

    fe = FeatureEngineer(
        use_whois=use_whois,
        use_abuseipdb=use_abuseipdb,
        use_greynoise=use_greynoise,
        abuseipdb_key=abuseipdb_key,
        greynoise_key=greynoise_key
    )

    df_features = fe.extract_features(df)

    print(f"Saving features to {output_path}...")
    df_features.to_parquet(output_path, index=False)
    print(f"✅ Done: {len(df_features):,} rows saved")

    return df_features


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 3:
        print("Usage: python feature_engineering.py <input.jsonl> <output.parquet>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    extract_features_from_jsonl(input_file, output_file)
