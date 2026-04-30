#!/usr/bin/env python3
"""
Ingestion orchestrator.

Runs ct_poller and phishtank labeling in a loop until a minimum number of
confirmed phishing instances are collected.

Thresholds (edit below):
    MIN_PHISHING_COUNT   — stop when we have at least this many phishing domains
    MIN_PHISHING_RATE    — also enforce a minimum rate (set to 0.0 to ignore)
    BATCH_DURATION_SECS  — how long to run ct_poller per iteration

Usage:
    python orchestrate.py

Both conditions must be met to stop:
    phishing_count >= MIN_PHISHING_COUNT
    phishing_rate  >= MIN_PHISHING_RATE  (if > 0)
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

# ── Thresholds ────────────────────────────────────────────────────────────────

MIN_PHISHING_COUNT  = 500      # minimum confirmed phishing domain instances in cert data
MIN_PHISHING_RATE   = 0.0005   # 0.05% — set to 0.0 to rely on count alone
BATCH_DURATION_SECS = 300      # run ct_poller for 5 min per iteration
MAX_ITERATIONS      = 20       # hard stop — won't run forever if PhishTank is stale

CERTS_FILE   = "certs_fallback.jsonl"
LABELS_FILE  = "phishtank_tranco_labels.jsonl"

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("orchestrate")

# ── Inlined CT poller (runs for a fixed duration then returns) ────────────────

def run_ct_poller(duration_secs: int) -> int:
    """
    Poll the CT log for `duration_secs` seconds, appending to CERTS_FILE.
    Returns number of certs written this batch.
    """
    import base64, signal, struct
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend

    CT_LOG_URL  = "https://ct.googleapis.com/logs/us1/argon2026h1/ct/v1"
    BATCH_SIZE  = 256
    POLL_DELAY  = 1.0

    def get_tree_size():
        r = requests.get(f"{CT_LOG_URL}/get-sth", timeout=10)
        r.raise_for_status()
        return r.json()["tree_size"]

    def get_entries(start, end):
        r = requests.get(f"{CT_LOG_URL}/get-entries",
                         params={"start": start, "end": end}, timeout=30)
        r.raise_for_status()
        return r.json().get("entries", [])

    def parse_entry(entry, index):
        try:
            leaf = base64.b64decode(entry["leaf_input"])
            entry_type = int.from_bytes(leaf[10:12], "big")
            if entry_type == 1:
                extra = base64.b64decode(entry.get("extra_data", ""))
                if len(extra) < 3:
                    return None
                cert_len = int.from_bytes(extra[:3], "big")
                der = extra[3:3 + cert_len]
            else:
                cert_len = int.from_bytes(leaf[12:15], "big")
                der = leaf[15:15 + cert_len]

            cert = x509.load_der_x509_certificate(der, default_backend())
            try:
                san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
                domains = san.value.get_values_for_type(x509.DNSName)
            except x509.ExtensionNotFound:
                domains = []
            if not domains:
                try:
                    cn = cert.subject.get_attributes_for_oid(x509.oid.NameOID.COMMON_NAME)
                    domains = [cn[0].value] if cn else []
                except Exception:
                    return None
            if not domains:
                return None

            def attr(name, oid):
                try:
                    a = name.get_attributes_for_oid(oid)
                    return a[0].value if a else None
                except Exception:
                    return None

            oid = x509.oid.NameOID
            return {
                "schema_version": 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cert_index": index,
                "fingerprint": cert.fingerprint(cert.signature_hash_algorithm).hex(":").upper()
                               if cert.signature_hash_algorithm else None,
                "serial": str(cert.serial_number),
                "not_before": cert.not_valid_before_utc.timestamp(),
                "not_after":  cert.not_valid_after_utc.timestamp(),
                "domains": list(domains),
                "subject": {"CN": attr(cert.subject, oid.COMMON_NAME),
                            "O":  attr(cert.subject, oid.ORGANIZATION_NAME)},
                "issuer":  {"CN": attr(cert.issuer,  oid.COMMON_NAME),
                            "O":  attr(cert.issuer,  oid.ORGANIZATION_NAME),
                            "aggregated": cert.issuer.rfc4514_string()},
                "source": {"name": "Google 'Argon2026h1' log", "url": CT_LOG_URL},
            }
        except Exception:
            return None

    start   = get_tree_size() - BATCH_SIZE
    written = 0
    deadline = time.time() + duration_secs

    with open(CERTS_FILE, "a", buffering=1) as f:
        while time.time() < deadline:
            try:
                entries = get_entries(start, start + BATCH_SIZE - 1)
                if not entries:
                    time.sleep(5)
                    continue
                for i, entry in enumerate(entries):
                    record = parse_entry(entry, start + i)
                    if record:
                        f.write(json.dumps(record, default=str) + "\n")
                        written += 1
                start += len(entries)
                time.sleep(POLL_DELAY)
            except requests.RequestException as exc:
                log.warning("CT API error: %s — retrying in 10s", exc)
                time.sleep(10)

    return written


# ── Inlined PhishTank loader ───────────────────────────────────────────────────

def refresh_phishtank_labels() -> int:
    """
    Download PhishTank feed, write phishtank_labels.jsonl.
    Returns number of phishing domains written.
    """
    PHISHTANK_URL = "http://data.phishtank.com/data/online-valid.csv"
    CACHE_FILE    = "phishtank_cache.csv"

    try:
        log.info("Downloading PhishTank feed …")
        resp = requests.get(PHISHTANK_URL, timeout=60,
                            headers={"User-Agent": "phishtank-labeler/1.0"})
        resp.raise_for_status()
        with open(CACHE_FILE, "wb") as f:
            f.write(resp.content)
    except requests.RequestException as exc:
        log.warning("PhishTank download failed: %s — using cache if available", exc)
        if not Path(CACHE_FILE).exists():
            raise

    df = pd.read_csv(CACHE_FILE, usecols=["url", "verified", "online"])
    verified = df[(df["verified"] == "yes") & (df["online"] == "yes")]

    def extract_domain(url):
        try:
            return urlparse(url).netloc.lower().split(":")[0]
        except Exception:
            return ""

    label_ts = datetime.now(timezone.utc).isoformat()
    domain_to_url: dict[str, str] = {}
    for url in verified["url"].str.lower().str.strip():
        d = extract_domain(url)
        if d and d not in domain_to_url:
            domain_to_url[d] = url

    with open(LABELS_FILE, "w", buffering=1) as f:
        for domain, url in domain_to_url.items():
            f.write(json.dumps({
                "domain":       domain,
                "url":          url,
                "y":            1,
                "label_source": "phishtank",
                "label_ts":     label_ts,
            }) + "\n")

    log.info("PhishTank: %d phishing domains written to %s", len(domain_to_url), LABELS_FILE)
    return len(domain_to_url)


# ── Distribution check ────────────────────────────────────────────────────────

def check_distribution() -> tuple[int, int, float]:
    """
    Join certs with labels, return (total_domains, phishing_count, phishing_rate).
    """
    if not Path(CERTS_FILE).exists():
        return 0, 0, 0.0
    if not Path(LABELS_FILE).exists():
        return 0, 0, 0.0

    df_certs = pd.read_json(CERTS_FILE, lines=True)
    df_labels = pd.read_json(LABELS_FILE, lines=True)

    df_domains = (df_certs
                  .explode("domains")
                  .rename(columns={"domains": "domain"})
                  .dropna(subset=["domain"]))
    df_domains["domain"] = df_domains["domain"].str.lower().str.lstrip("*.")

    df = df_domains.merge(df_labels[["domain", "y"]], on="domain", how="left")
    df["y"] = df["y"].fillna(0).astype(int)

    total    = len(df)
    phishing = int(df["y"].sum())
    rate     = phishing / total if total else 0.0
    return total, phishing, rate


# ── Orchestration loop ────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Ingestion orchestrator starting")
    log.info("  Target: >= %d phishing instances AND >= %.3f%% rate",
             MIN_PHISHING_COUNT, MIN_PHISHING_RATE * 100)
    log.info("  Batch duration: %ds | Max iterations: %d",
             BATCH_DURATION_SECS, MAX_ITERATIONS)
    log.info("=" * 60)

    for iteration in range(1, MAX_ITERATIONS + 1):
        log.info("── Iteration %d/%d ──────────────────────────", iteration, MAX_ITERATIONS)

        # 1. Collect certs
        log.info("Running CT poller for %ds …", BATCH_DURATION_SECS)
        written = run_ct_poller(BATCH_DURATION_SECS)
        log.info("Batch complete — %d certs written to %s", written, CERTS_FILE)

        # 2. Refresh labels
        refresh_phishtank_labels()

        # 3. Check distribution
        total, phishing, rate = check_distribution()
        log.info(
            "Distribution — total:%d phishing:%d rate:%.4f%% | "
            "target_count:%d target_rate:%.3f%%",
            total, phishing, rate * 100,
            MIN_PHISHING_COUNT, MIN_PHISHING_RATE * 100,
        )

        # 4. Evaluate thresholds
        count_ok = phishing >= MIN_PHISHING_COUNT
        rate_ok  = (MIN_PHISHING_RATE == 0.0) or (rate >= MIN_PHISHING_RATE)

        if count_ok and rate_ok:
            log.info("✓ Thresholds met — saving dataset and stopping.")
            log.info("  Final: %d total domains | %d phishing (%.4f%%)",
                     total, phishing, rate * 100)
            break

        remaining_count = max(0, MIN_PHISHING_COUNT - phishing)
        log.info(
            "✗ Thresholds not met — need %d more phishing instances. "
            "Continuing …", remaining_count,
        )

        if iteration == MAX_ITERATIONS:
            log.warning(
                "Max iterations reached. Stopping with %d phishing instances (%.4f%%). "
                "Consider lowering MIN_PHISHING_COUNT or extending BATCH_DURATION_SECS.",
                phishing, rate * 100,
            )

    # Final summary
    total, phishing, rate = check_distribution()
    print("\n── Final dataset ────────────────────────────────────────")
    print(f"  Certs file:   {CERTS_FILE}")
    print(f"  Labels file:  {LABELS_FILE}")
    print(f"  Total domains: {total:,}")
    print(f"  Phishing (y=1): {phishing:,}  ({rate*100:.4f}%)")
    print(f"  Unknown  (y=0): {total - phishing:,}")
    print("─────────────────────────────────────────────────────────")
    print("\nLoad in notebook:")
    print("  import pandas as pd")
    print(f"  df_certs  = pd.read_json('{CERTS_FILE}', lines=True)")
    print(f"  df_labels = pd.read_json('{LABELS_FILE}', lines=True)")


if __name__ == "__main__":
    main()