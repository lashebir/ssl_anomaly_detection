"""
PhishTank ground truth loader.

Provides a PhishTankLabeler class that:
  1. Downloads the PhishTank verified phish CSV (updated hourly by PhishTank)
  2. Parses out domains from URLs
  3. Exposes a label(domain) method → 1 (phishing), 0 (unknown/legitimate)
  4. Auto-refreshes the dataset on a configurable interval

Usage:
    labeler = PhishTankLabeler()
    labeler.refresh()               # download now

    label = labeler.label("paypa1-secure-login.com")   # → 1 or 0
    df["y"] = df["domain"].apply(labeler.label)

Register for a free API key at https://www.phishtank.com/api_info.php
Set PHISHTANK_API_KEY env var or pass it directly to the constructor.
Without a key you can still download the public feed, but you'll hit
rate limits quickly — the key just unlocks higher limits.

── Tranco legitimate domain list ─────────────────────────────────────────────

PhishTank gives you positive labels (phishing = 1).
For negative labels (legitimate = 0) you need a separate source.
The Tranco list ranks the top 1M domains by traffic — overwhelmingly legitimate.

Use it to build a clean negative class:
  - Sample N domains from Tranco top 10k (very high confidence legitimate)
  - Label them 0
  - Combine with PhishTank positives (label 1)
  - Your training set is this combined labeled pool

Without Tranco negatives, you only have:
  phishing=1  from PhishTank
  unknown=0   from everything else in your cert stream
...which means your "0" class is noisy (some real phishing slips through unlabeled).
That's fine for a first model but will hurt precision.

"""

import logging
import os
import time
import json
from datetime import datetime, timezone
from urllib.parse import urlparse

import pandas as pd
import requests

log = logging.getLogger("phishtank")

# ── Config ────────────────────────────────────────────────────────────────────

PHISHTANK_URL = (
    "http://data.phishtank.com/data/{key}online-valid.csv"
)
REFRESH_INTERVAL_SECONDS = 3600          # PhishTank updates hourly; match it
CACHE_FILE = "phishtank_cache.csv"      # avoid re-downloading if process restarts


class PhishTankLabeler:
    def __init__(
        self,
        api_key: str = "",
        refresh_interval: int = REFRESH_INTERVAL_SECONDS,
        cache_file: str = CACHE_FILE,
    ):
        self.api_key = api_key or os.getenv("PHISHTANK_API_KEY", "")
        self.refresh_interval = refresh_interval
        self.cache_file = cache_file
        self._phishing_domains: set[str] = set()
        self._phishing_urls: set[str] = set()
        self._last_refresh: float = 0
        self._total_entries: int = 0

    # ── Download ──────────────────────────────────────────────────────────────

    def refresh(self, force: bool = False) -> None:
        """
        Download fresh PhishTank data if the refresh interval has elapsed.
        Pass force=True to download regardless.
        """
        if not force and (time.time() - self._last_refresh) < self.refresh_interval:
            log.debug("PhishTank data is fresh, skipping refresh")
            return

        key_segment = f"{self.api_key}/" if self.api_key else ""
        url = PHISHTANK_URL.format(key=key_segment)
        log.info("Downloading PhishTank feed: %s", url)

        try:
            resp = requests.get(url, timeout=60, headers={"User-Agent": "phishtank-labeler/1.0"})
            resp.raise_for_status()
            with open(self.cache_file, "wb") as f:
                f.write(resp.content)
            log.info("PhishTank feed saved to %s (%d bytes)", self.cache_file, len(resp.content))
        except requests.RequestException as exc:
            log.warning("PhishTank download failed: %s — trying cache", exc)
            if not os.path.exists(self.cache_file):
                raise RuntimeError("No PhishTank cache available and download failed") from exc

        self._load_cache()
        self._last_refresh = time.time()

    def _load_cache(self) -> None:
        """Parse the cached CSV and build domain + URL lookup sets."""
        df = pd.read_csv(self.cache_file, usecols=["url", "verified", "online"])

        # PhishTank CSV columns: phish_id, url, phish_detail_url, submission_time,
        # verified, verification_time, online, target
        # Keep only verified + currently online phishes for highest-confidence labels
        verified = df[(df["verified"] == "yes") & (df["online"] == "yes")]

        self._phishing_urls = set(verified["url"].str.lower().str.strip())
        self._phishing_domains = set(
            self._extract_domain(u) for u in self._phishing_urls
            if self._extract_domain(u)
        )
        self._total_entries = len(verified)
        log.info(
            "PhishTank loaded — %d verified+online entries, %d unique domains",
            self._total_entries, len(self._phishing_domains),
        )

    # ── Labeling ──────────────────────────────────────────────────────────────

    def label(self, domain: str) -> int:
        """
        Returns:
            1  — domain appears in PhishTank verified phishing list
            0  — not found (treat as unknown/legitimate for now)

        Note: 0 does NOT mean definitively legitimate. PhishTank only covers
        reported phishing. Use a legitimate domain list (Tranco) for true
        negative labels — see note at bottom of file.
        """
        self._maybe_refresh()
        domain = domain.lower().strip().lstrip("*.")  # strip wildcard subdomains
        return 1 if domain in self._phishing_domains else 0

    def label_url(self, url: str) -> int:
        """Label by full URL — higher precision than domain-only lookup."""
        self._maybe_refresh()
        return 1 if url.lower().strip() in self._phishing_urls else 0

    def label_dataframe(self, df: pd.DataFrame, domain_col: str = "domain") -> pd.Series:
        """
        Vectorised labeling for a full dataframe.
        Returns a Series of 0/1 aligned to df's index.

            df["y"] = labeler.label_dataframe(df)
        """
        self._maybe_refresh()
        return df[domain_col].str.lower().str.strip().str.lstrip("*.").map(
            lambda d: 1 if d in self._phishing_domains else 0
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _maybe_refresh(self) -> None:
        if (time.time() - self._last_refresh) > self.refresh_interval:
            try:
                self.refresh()
            except Exception as exc:
                log.warning("Background PhishTank refresh failed: %s", exc)

    @staticmethod
    def _extract_domain(url: str) -> str:
        try:
            return urlparse(url).netloc.lower().split(":")[0]  # strip port
        except Exception:
            return ""

    @property
    def stats(self) -> dict:
        return {
            "phishing_domains": len(self._phishing_domains),
            "phishing_urls": len(self._phishing_urls),
            "total_entries": self._total_entries,
            "last_refresh": datetime.fromtimestamp(self._last_refresh, tz=timezone.utc).isoformat()
            if self._last_refresh else None,
        }

def load_tranco(n: int = 10_000, cache_file: str = "tranco_top1m.csv") -> set[str]:
    """
    Download and return the top-n Tranco domains as a set.
    https://tranco-list.eu
    """
    if not os.path.exists(cache_file):
        log.info("Downloading Tranco top-1M list …")
        resp = requests.get("https://tranco-list.eu/top-1m.csv.zip", timeout=60)
        resp.raise_for_status()
        import io, zipfile
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f, header=None, names=["rank", "domain"])
        df.to_csv(cache_file, index=False)
        log.info("Tranco saved to %s", cache_file)
    else:
        df = pd.read_csv(cache_file)

    top_n = set(df.head(n)["domain"].str.lower())
    log.info("Tranco loaded — top %d domains", len(top_n))
    return top_n

def write_labels_jsonl(
    labeler: "PhishTankLabeler",
    output_file: str = "phishtank_labels.jsonl",
) -> None:
    """
    Write one record per known-phishing domain to a JSONL file.
    Schema:
        domain        — the phishing domain (str)
        url           — one representative phishing URL for that domain (str)
        y             — label, always 1 for this file (int)
        label_source  — "phishtank" (str)
        label_ts      — UTC ISO-8601 timestamp of when this file was written (str)
 
    In your notebook, join on domain:
        labels = pd.read_json("phishtank_labels.jsonl", lines=True)
        df = df.merge(labels[["domain","y","label_source"]], on="domain", how="left")
        df["y"] = df["y"].fillna(0).astype(int)   # 0 = not in PhishTank
    """
    label_ts = datetime.now(timezone.utc).isoformat()
 
    # Build a domain → url mapping from the raw URL set for traceability
    domain_to_url: dict[str, str] = {}
    for url in labeler._phishing_urls:
        domain = labeler._extract_domain(url)
        if domain and domain not in domain_to_url:
            domain_to_url[domain] = url
 
    written = 0
    with open(output_file, "w", buffering=1) as f:
        for domain, url in domain_to_url.items():
            record = {
                "domain":       domain,
                "url":          url,
                "y":            1,
                "label_source": "phishtank",
                "label_ts":     label_ts,
            }
            f.write(json.dumps(record) + "\n")
            written += 1
 
    log.info("Wrote %d phishing labels to %s", written, output_file)


# ── EDA usage example ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # 1. Load PhishTank labels
    labeler = PhishTankLabeler()
    labeler.refresh()
    print(labeler.stats)

    # Write labels to their own JSONL — join with cert data in the notebook
    write_labels_jsonl(labeler, output_file="phishtank_labels.jsonl")


    # 4. Optionally build a balanced set with Tranco negatives
    # tranco = load_tranco(n=10_000)
    # df_legit = pd.DataFrame({"domain": list(tranco), "y": 0})
    # df_labeled = pd.concat([df_domains[df_domains["y"] == 1], df_legit])