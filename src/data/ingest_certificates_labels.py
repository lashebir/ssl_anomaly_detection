#!/usr/bin/env python3
"""
Unified SSL certificate and phishing label ingestion pipeline.

Combines six ingestion modes into a single script with configurable outputs:
  1. live                        — Poll CT log + fetch labels for collected domains (integrated)
  2. live-certs                  — Poll Google CT log for live certificates only
  3. live-labels                 — Fetch PhishTank labels for domains in a cert file
  4. historical-phishing-labels  — Import all PhishTank phishing domains as labels
  5. historical-phishing-certs   — Fetch historical certs for known phishing domains via crt.sh
  6. windows                     — Fetch CT log windows around phishing cert timestamps

Usage examples:
  # Integrated live pipeline (recommended) - collects certs then labels domains
  python ingest_certificates_labels.py live --certs certs.jsonl --labels labels.jsonl --duration 900

  # Individual modes
  python ingest_certificates_labels.py live-certs --output certs.jsonl --duration 900
  python ingest_certificates_labels.py live-labels --certs certs.jsonl --output labels.jsonl
  python ingest_certificates_labels.py historical-phishing-labels --output phishing_labels.jsonl
  python ingest_certificates_labels.py historical-phishing-certs --labels phishing_labels.jsonl --output hist.jsonl
  python ingest_certificates_labels.py windows --phishing hist.jsonl --output windows.jsonl

Install:
  pip install requests cryptography pandas

Environment:
  PHISHTANK_API_KEY — optional, increases rate limits (get from phishtank.com/api_info.php)
"""

import argparse
import base64
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from cryptography import x509
from cryptography.hazmat.backends import default_backend

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("ingest")

# ══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

_running = True
_output_file = None


def shutdown_handler(sig, frame):
    """Clean shutdown on SIGINT/SIGTERM."""
    global _running
    log.info("Shutting down …")
    _running = False
    if _output_file and not _output_file.closed:
        _output_file.close()
    sys.exit(0)


def parse_ct_entry(entry: dict, index: int) -> dict | None:
    """
    Parse a CT log entry (MerkleTreeLeaf) into normalized cert record.
    Handles both X509LogEntry (type 0) and PrecertLogEntry (type 1).

    Returns None if parsing fails or cert has no domains.
    """
    try:
        leaf_input = base64.b64decode(entry["leaf_input"])

        # CT MerkleTreeLeaf: 2 bytes header + 8 bytes timestamp + 2 bytes entry_type
        entry_type = int.from_bytes(leaf_input[10:12], "big")

        if entry_type == 1:
            # Precert — extract from extra_data
            extra = base64.b64decode(entry.get("extra_data", ""))
            if len(extra) < 3:
                return None
            cert_len = int.from_bytes(extra[:3], "big")
            der = extra[3:3 + cert_len]
        else:
            # X509 — cert follows 3-byte length at offset 12
            cert_len = int.from_bytes(leaf_input[12:15], "big")
            der = leaf_input[15:15 + cert_len]

        cert = x509.load_der_x509_certificate(der, default_backend())

        # Extract SANs
        try:
            san_ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
            domains = san_ext.value.get_values_for_type(x509.DNSName)
        except x509.ExtensionNotFound:
            domains = []

        # Fallback to CN if no SANs
        if not domains:
            try:
                cn = cert.subject.get_attributes_for_oid(x509.oid.NameOID.COMMON_NAME)
                domains = [cn[0].value] if cn else []
            except Exception:
                domains = []

        if not domains:
            return None

        def name_attr(name, oid):
            try:
                attrs = name.get_attributes_for_oid(oid)
                return attrs[0].value if attrs else None
            except Exception:
                return None

        oid = x509.oid.NameOID
        issuer = cert.issuer
        subject = cert.subject

        return {
            "schema_version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cert_index": index,
            "fingerprint": cert.fingerprint(cert.signature_hash_algorithm).hex(":").upper()
                           if cert.signature_hash_algorithm else None,
            "serial": str(cert.serial_number),
            "not_before": cert.not_valid_before_utc.timestamp(),
            "not_after": cert.not_valid_after_utc.timestamp(),
            "domains": list(domains),
            "subject": {
                "CN": name_attr(subject, oid.COMMON_NAME),
                "O": name_attr(subject, oid.ORGANIZATION_NAME),
                "C": name_attr(subject, oid.COUNTRY_NAME),
            },
            "issuer": {
                "CN": name_attr(issuer, oid.COMMON_NAME),
                "O": name_attr(issuer, oid.ORGANIZATION_NAME),
                "C": name_attr(issuer, oid.COUNTRY_NAME),
                "aggregated": issuer.rfc4514_string(),
            },
            "source": {
                "name": "Google 'Argon2026h1' log",
                "url": "https://ct.googleapis.com/logs/us1/argon2026h1/ct/v1",
            },
        }

    except Exception as exc:
        log.debug("Parse error at index %d: %s", index, exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# MODE 0: INTEGRATED LIVE PIPELINE (CERTS + LABELS)
# ══════════════════════════════════════════════════════════════════════════════

def run_live_pipeline(args):
    """
    Integrated pipeline that collects live certs and fetches labels for those domains.
    Runs for multiple iterations with configurable batch duration.
    """
    global _running

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    log.info("Starting live pipeline — %d iterations × %ds", args.iterations, args.duration)

    for iteration in range(1, args.iterations + 1):
        if not _running:
            break

        log.info("")
        log.info("── Iteration %d/%d ──────────────────────────────────", iteration, args.iterations)

        # Step 1: Collect certs for the batch duration
        log.info("[1/2] Collecting certs for %ds …", args.duration)

        # Create a temporary args object for live_certs
        cert_args = argparse.Namespace(
            output=args.certs,
            ct_log_url=args.ct_log_url,
            batch_size=args.batch_size,
            poll_delay=args.poll_delay,
            start_index=None,
            duration=args.duration,
        )

        try:
            run_live_certs(cert_args)
        except KeyboardInterrupt:
            log.info("Cert collection interrupted by user")
            break
        except Exception as exc:
            log.warning("Cert collection failed: %s", exc)

        if not _running:
            break

        # Step 2: Fetch labels for domains in the collected certs
        log.info("[2/2] Fetching labels for collected cert domains …")

        label_args = argparse.Namespace(
            certs=args.certs,
            output=args.labels,
            api_key=args.api_key,
            phishtank_cache=args.phishtank_cache,
            tranco_cache=args.tranco_cache,
        )

        try:
            run_live_labels(label_args)
        except Exception as exc:
            log.warning("Label fetch failed: %s", exc)

        log.info("Iteration %d complete — %s", iteration, datetime.now(timezone.utc).isoformat())

    log.info("")
    log.info("Live pipeline complete. Files written:")
    for filepath in [args.certs, args.labels]:
        if Path(filepath).exists():
            try:
                with open(filepath) as f:
                    lines = sum(1 for _ in f)
                log.info("  %s: %d lines", filepath, lines)
            except Exception:
                log.info("  %s: exists", filepath)


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1: LIVE CT LOG POLLING
# ══════════════════════════════════════════════════════════════════════════════

def run_live_certs(args):
    """Poll CT log and write certificates to JSONL in real-time."""
    global _output_file, _running

    # Only set signal handlers if not already set (avoid conflicts with pipeline mode)
    if args.duration is None:
        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

    CT_LOG_URL = args.ct_log_url
    BATCH_SIZE = args.batch_size
    POLL_DELAY = args.poll_delay

    stats = {"fetched": 0, "parsed": 0, "errors": 0, "start_time": time.time()}

    def print_stats():
        elapsed = time.time() - stats["start_time"]
        rate = stats["fetched"] / elapsed if elapsed else 0
        log.info("fetched:%d parsed:%d errors:%d rate:%.1f/s",
                 stats["fetched"], stats["parsed"], stats["errors"], rate)

    # Get current tree size
    resp = requests.get(f"{CT_LOG_URL}/get-sth", timeout=10)
    resp.raise_for_status()
    tree_size = resp.json()["tree_size"]
    log.info("CT log tree size: %d", tree_size)

    start = args.start_index if args.start_index is not None else tree_size - BATCH_SIZE
    log.info("Starting poll from index %d (batch=%d)", start, BATCH_SIZE)

    _output_file = open(args.output, "a", buffering=1)
    log.info("Writing to %s", args.output)

    start_time = time.time()

    while _running:
        # Stop if duration limit reached
        if args.duration and (time.time() - start_time) >= args.duration:
            log.info("Duration limit reached (%ds)", args.duration)
            break

        try:
            end = start + BATCH_SIZE - 1
            resp = requests.get(
                f"{CT_LOG_URL}/get-entries",
                params={"start": start, "end": end},
                timeout=30,
            )
            resp.raise_for_status()
            entries = resp.json().get("entries", [])

            if not entries:
                log.info("No new entries at index %d, waiting …", start)
                time.sleep(5)
                continue

            for i, entry in enumerate(entries):
                stats["fetched"] += 1
                record = parse_ct_entry(entry, start + i)
                if record:
                    record["data_source"] = "argon_live"
                    stats["parsed"] += 1
                    _output_file.write(json.dumps(record, default=str) + "\n")
                else:
                    stats["errors"] += 1

            start += len(entries)

            if stats["fetched"] % 2_000 == 0:
                print_stats()

            time.sleep(POLL_DELAY)

        except requests.RequestException as exc:
            log.warning("API error: %s — retrying in 10s", exc)
            time.sleep(10)

    print_stats()
    _output_file.close()
    _output_file = None
    log.info("Live cert ingestion complete")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2: LIVE PHISHTANK LABELS
# ══════════════════════════════════════════════════════════════════════════════

def run_live_labels(args):
    """
    Extract domains from cert JSONL and fetch labels from PhishTank.
    Only writes labels for domains found in the cert file.
    """
    PHISHTANK_URL = "http://data.phishtank.com/data/{key}online-valid.csv"

    # Step 1: Extract unique domains from cert file
    if not Path(args.certs).exists():
        log.error("Cert file not found: %s", args.certs)
        raise FileNotFoundError(f"Cert file not found: {args.certs}")

    log.info("Extracting domains from %s …", args.certs)
    cert_domains = set()

    with open(args.certs, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
                domains = record.get("domains", [])
                for domain in domains:
                    # Normalize: lowercase, strip wildcards
                    normalized = domain.lower().strip().lstrip("*.")
                    if normalized:
                        cert_domains.add(normalized)
            except json.JSONDecodeError:
                continue

    log.info("Found %d unique domains in cert file", len(cert_domains))

    if not cert_domains:
        log.warning("No domains found in cert file — skipping label fetch")
        return

    # Step 2: Download/load PhishTank data
    api_key = args.api_key or os.getenv("PHISHTANK_API_KEY", "")
    key_segment = f"{api_key}/" if api_key else ""
    url = PHISHTANK_URL.format(key=key_segment)
    phishtank_cache_file = args.phishtank_cache if hasattr(args, 'phishtank_cache') else "phishtank_cache.csv"

    log.info("Downloading PhishTank feed: %s", url)

    try:
        resp = requests.get(url, timeout=60, headers={"User-Agent": "phishtank-labeler/1.0"})
        resp.raise_for_status()
        with open(phishtank_cache_file, "wb") as f:
            f.write(resp.content)
        log.info("PhishTank feed saved to %s (%d bytes)", phishtank_cache_file, len(resp.content))
    except requests.RequestException as exc:
        log.warning("PhishTank download failed: %s", exc)
        if not api_key:
            log.info("Tip: Register for a free API key at https://phishtank.com/api_info.php")
            log.info("Then set PHISHTANK_API_KEY env var or use --api-key argument")

        if not Path(phishtank_cache_file).exists():
            raise RuntimeError(
                f"No PhishTank cache available at {phishtank_cache_file} and download failed. "
                f"Cannot proceed without labels. "
                f"Try downloading manually from https://phishtank.com/developer_info.php"
            ) from exc

        # Show cache age
        cache_age = time.time() - Path(phishtank_cache_file).stat().st_mtime
        cache_age_hours = cache_age / 3600
        log.info("Using cached PhishTank data from %s (%.1f hours old)", phishtank_cache_file, cache_age_hours)

    # Step 3: Parse PhishTank CSV and build lookup
    df = pd.read_csv(phishtank_cache_file, usecols=["url", "verified", "online"])
    verified = df[(df["verified"] == "yes") & (df["online"] == "yes")]

    log.info("PhishTank loaded — %d verified+online entries", len(verified))

    # Build domain -> URL mapping from PhishTank
    def extract_domain(url: str) -> str:
        try:
            return urlparse(url).netloc.lower().split(":")[0]
        except Exception:
            return ""

    phishing_domains = {}  # domain -> url
    for phish_url in verified["url"].str.lower().str.strip():
        domain = extract_domain(phish_url)
        if domain and domain not in phishing_domains:
            phishing_domains[domain] = phish_url

    log.info("PhishTank contains %d unique phishing domains", len(phishing_domains))

    # Step 4: Load Tranco top-10k for legitimate domain labels
    tranco_cache = args.tranco_cache if hasattr(args, 'tranco_cache') else "tranco_top1m.csv"

    if not Path(tranco_cache).exists():
        log.info("Downloading Tranco top-1M list …")
        try:
            resp = requests.get("https://tranco-list.eu/top-1m.csv.zip", timeout=60)
            resp.raise_for_status()
            import io, zipfile
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    tranco_df = pd.read_csv(f, header=None, names=["rank", "domain"])
            tranco_df.to_csv(tranco_cache, index=False)
            log.info("Tranco saved to %s", tranco_cache)
        except Exception as exc:
            log.warning("Tranco download failed: %s — proceeding without legitimate labels", exc)
            tranco_df = pd.DataFrame(columns=["rank", "domain"])
    else:
        tranco_df = pd.read_csv(tranco_cache)

    tranco_top10k = set(tranco_df.head(10_000)["domain"].str.lower())
    log.info("Tranco loaded — %d top domains for legitimate labels", len(tranco_top10k))

    # Step 5: Label all cert domains
    label_ts = datetime.now(timezone.utc).isoformat()
    labels = []
    phishing_count = 0
    legitimate_count = 0
    unknown_count = 0

    for cert_domain in cert_domains:
        if cert_domain in phishing_domains:
            # High confidence phishing from PhishTank
            labels.append({
                "domain": cert_domain,
                "url": phishing_domains[cert_domain],
                "y": 1,
                "label_source": "phishtank",
                "label_ts": label_ts,
            })
            phishing_count += 1
        elif cert_domain in tranco_top10k:
            # High confidence legitimate from Tranco top-10k
            labels.append({
                "domain": cert_domain,
                "url": None,
                "y": 0,
                "label_source": "tranco",
                "label_ts": label_ts,
            })
            legitimate_count += 1
        else:
            # Unknown — not in PhishTank or Tranco top-10k
            labels.append({
                "domain": cert_domain,
                "url": None,
                "y": 0,
                "label_source": "unknown",
                "label_ts": label_ts,
            })
            unknown_count += 1

    log.info("Labeled %d domains: %d phishing, %d legitimate (Tranco), %d unknown",
             len(labels), phishing_count, legitimate_count, unknown_count)

    # Step 6: Append all labels to JSONL (preserves existing data)
    with open(args.output, "a", buffering=1) as f:
        for record in labels:
            f.write(json.dumps(record) + "\n")

    log.info("Wrote %d labels to %s", len(labels), args.output)


# ══════════════════════════════════════════════════════════════════════════════
# MODE 4: HISTORICAL PHISHING LABELS
# ══════════════════════════════════════════════════════════════════════════════

def run_historical_phishing_labels(args):
    """
    Download PhishTank and write all phishing domains as labels.
    Output can be used as input to mode 5 (historical-phishing-certs) to fetch certs.
    """
    PHISHTANK_URL = "http://data.phishtank.com/data/{key}online-valid.csv"

    # Step 1: Download/load PhishTank data
    api_key = args.api_key or os.getenv("PHISHTANK_API_KEY", "")
    key_segment = f"{api_key}/" if api_key else ""
    url = PHISHTANK_URL.format(key=key_segment)
    phishtank_cache_file = args.phishtank_cache

    log.info("Downloading PhishTank feed: %s", url)

    try:
        resp = requests.get(url, timeout=60, headers={"User-Agent": "phishtank-labeler/1.0"})
        resp.raise_for_status()
        with open(phishtank_cache_file, "wb") as f:
            f.write(resp.content)
        log.info("PhishTank feed saved to %s (%d bytes)", phishtank_cache_file, len(resp.content))
    except requests.RequestException as exc:
        log.warning("PhishTank download failed: %s", exc)
        if not api_key:
            log.info("Tip: Register for a free API key at https://phishtank.com/api_info.php")
            log.info("Then set PHISHTANK_API_KEY env var or use --api-key argument")

        if not Path(phishtank_cache_file).exists():
            raise RuntimeError(
                f"No PhishTank cache available at {phishtank_cache_file} and download failed. "
                f"Cannot proceed without labels. "
                f"Try downloading manually from https://phishtank.com/developer_info.php"
            ) from exc

        # Show cache age
        cache_age = time.time() - Path(phishtank_cache_file).stat().st_mtime
        cache_age_hours = cache_age / 3600
        log.info("Using cached PhishTank data from %s (%.1f hours old)", phishtank_cache_file, cache_age_hours)

    # Step 2: Parse PhishTank CSV
    df = pd.read_csv(phishtank_cache_file, usecols=["url", "verified", "online"])
    verified = df[(df["verified"] == "yes") & (df["online"] == "yes")]

    log.info("PhishTank loaded — %d verified+online entries", len(verified))

    # Step 3: Extract domains from URLs
    def extract_domain(url: str) -> str:
        try:
            return urlparse(url).netloc.lower().split(":")[0]
        except Exception:
            return ""

    phishing_labels = []
    label_ts = datetime.now(timezone.utc).isoformat()

    domain_to_url = {}
    for phish_url in verified["url"].str.lower().str.strip():
        domain = extract_domain(phish_url)
        if domain and domain not in domain_to_url:
            domain_to_url[domain] = phish_url

    log.info("Extracted %d unique phishing domains", len(domain_to_url))

    # Step 4: Create label records
    for domain, url in domain_to_url.items():
        phishing_labels.append({
            "domain": domain,
            "url": url,
            "y": 1,
            "label_source": "phishtank",
            "label_ts": label_ts,
        })

    # Step 5: Write to output file (default: append)
    mode = "w" if args.overwrite else "a"
    with open(args.output, mode, buffering=1) as f:
        for record in phishing_labels:
            f.write(json.dumps(record) + "\n")

    log.info("Wrote %d phishing labels to %s (mode: %s)", len(phishing_labels), args.output,
             "overwrite" if args.overwrite else "append")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 5: HISTORICAL PHISHING CERTS VIA CRT.SH
# ══════════════════════════════════════════════════════════════════════════════

def run_historical_phishing_certs(args):
    """Fetch historical certs for known phishing domains from crt.sh."""
    if not Path(args.labels).exists():
        log.error("Labels file not found: %s", args.labels)
        sys.exit(1)

    labels_df = pd.read_json(args.labels, lines=True)
    domains = labels_df["domain"].dropna().unique().tolist()

    if args.max_domains:
        domains = domains[:args.max_domains]

    log.info("Fetching crt.sh certs for %d phishing domains → %s", len(domains), args.output)

    label_ts = labels_df["label_ts"].iloc[0] if "label_ts" in labels_df.columns else \
               datetime.now(timezone.utc).isoformat()

    stats = {"domains_queried": 0, "certs_written": 0, "domains_empty": 0}
    seen_serials = set()

    with open(args.output, "a", buffering=1) as out:
        for i, domain in enumerate(domains):
            raw_certs = fetch_crtsh_certs(domain, args.max_certs_per_domain)
            stats["domains_queried"] += 1

            if not raw_certs:
                stats["domains_empty"] += 1
            else:
                for raw in raw_certs:
                    serial = raw.get("serial_number", "")
                    if serial and serial in seen_serials:
                        continue

                    record = normalize_crtsh_record(raw, domain, label_ts)
                    if record:
                        out.write(json.dumps(record, default=str) + "\n")
                        stats["certs_written"] += 1
                        if serial:
                            seen_serials.add(serial)

            if (i + 1) % 50 == 0:
                log.info("Progress: %d/%d domains | %d certs written | %d empty",
                         i + 1, len(domains), stats["certs_written"], stats["domains_empty"])

            time.sleep(args.request_delay)

    log.info("Historical cert ingestion complete — %s", stats)


def fetch_crtsh_certs(domain: str, max_certs: int) -> list[dict]:
    """Query crt.sh for certificates issued to a domain."""
    try:
        resp = requests.get(
            "https://crt.sh/json",
            params={"q": domain, "output": "json"},
            timeout=30,
            headers={"Accept": "application/json"},
        )
        if resp.status_code == 429:
            log.warning("Rate limited by crt.sh — sleeping 30s")
            time.sleep(30)
            return []
        if resp.status_code != 200:
            log.warning("crt.sh returned %d for %s", resp.status_code, domain)
            return []
        return resp.json()[:max_certs]
    except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
        log.warning("crt.sh request failed for %s: %s", domain, exc)
        return []


def normalize_crtsh_record(raw: dict, phishing_domain: str, label_ts: str) -> dict | None:
    """Normalize crt.sh record to common cert schema."""
    try:
        # Parse SANs
        raw_names = raw.get("name_value", "") or ""
        domains = [
            d.strip().lower()
            for d in raw_names.splitlines()
            if d.strip() and not d.strip().startswith("?")
        ]
        if not domains:
            cn = raw.get("common_name", "")
            domains = [cn.lower()] if cn else []
        if not domains:
            return None

        # Parse timestamps
        def parse_ts(s):
            if not s:
                return None
            try:
                return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=timezone.utc).timestamp()
            except ValueError:
                return None

        # Parse issuer DN
        issuer_str = raw.get("issuer_name", "") or ""
        issuer_parts = dict(
            part.strip().split("=", 1)
            for part in issuer_str.split(",")
            if "=" in part
        )

        return {
            "schema_version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cert_index": None,
            "fingerprint": None,
            "serial": raw.get("serial_number"),
            "not_before": parse_ts(raw.get("not_before")),
            "not_after": parse_ts(raw.get("not_after")),
            "domains": domains,
            "subject": {
                "CN": raw.get("common_name"),
                "O": None,
            },
            "issuer": {
                "CN": issuer_parts.get("CN"),
                "O": issuer_parts.get("O"),
                "C": issuer_parts.get("C"),
                "aggregated": issuer_str,
            },
            "source": {
                "name": "crt.sh historical lookup",
                "url": f"https://crt.sh/?id={raw.get('id', '')}",
            },
            "data_source": "crtsh_historical",
            "phishing_domain": phishing_domain,
            "label_source": "phishtank",
            "label_ts": label_ts,
            "y": 1,
        }
    except Exception as exc:
        log.debug("Normalization failed: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# MODE 6: CT HISTORICAL WINDOWS
# ══════════════════════════════════════════════════════════════════════════════

def run_windows(args):
    """Fetch CT log windows around phishing cert timestamps."""
    if not Path(args.phishing).exists():
        log.error("Phishing certs file not found: %s", args.phishing)
        sys.exit(1)

    df_phish = pd.read_json(args.phishing, lines=True)

    # Select anchors
    df_anchors = (df_phish
                  .dropna(subset=["not_before", "domains"])
                  .drop_duplicates(subset=["serial"])
                  .head(args.max_anchors))

    log.info("Using %d phishing certs as time series anchors", len(df_anchors))

    # Get CT log tree size
    CT_LOG_URL = args.ct_log_url
    resp = requests.get(f"{CT_LOG_URL}/get-sth", timeout=10)
    resp.raise_for_status()
    tree_size = resp.json()["tree_size"]
    log.info("CT log tree size: %d", tree_size)

    stats = {"anchors_processed": 0, "certs_written": 0}

    with open(args.output, "a", buffering=1) as out:
        for _, row in df_anchors.iterrows():
            anchor_ts = row["not_before"]
            anchor_domain = row["domains"][0] if isinstance(row["domains"], list) else row["domains"]

            log.info("Finding CT index for %s (ts=%s) …",
                     anchor_domain,
                     datetime.fromtimestamp(anchor_ts, tz=timezone.utc).isoformat())

            anchor_index = timestamp_to_ct_index(anchor_ts, tree_size, CT_LOG_URL)
            log.info("  → index %d — fetching ±%d window", anchor_index, args.window_size)

            written = fetch_ct_window(
                anchor_index, anchor_domain, anchor_ts,
                tree_size, CT_LOG_URL, args.window_size, out
            )
            stats["certs_written"] += written
            stats["anchors_processed"] += 1

            log.info("  → wrote %d records", written)
            time.sleep(args.request_delay)

    log.info("Window ingestion complete — %s", stats)


def timestamp_to_ct_index(target_ts: float, tree_size: int, ct_log_url: str) -> int:
    """Binary search CT log to find index closest to target timestamp."""
    lo, hi = 0, tree_size - 1

    for _ in range(30):
        if lo >= hi:
            break
        mid = (lo + hi) // 2
        try:
            resp = requests.get(
                f"{ct_log_url}/get-entries",
                params={"start": mid, "end": mid},
                timeout=10,
            )
            resp.raise_for_status()
            entries = resp.json().get("entries", [])
            if not entries:
                break

            leaf = base64.b64decode(entries[0]["leaf_input"])
            ms = int.from_bytes(leaf[2:10], "big")
            entry_ts = ms / 1000.0

            if entry_ts < target_ts:
                lo = mid + 1
            else:
                hi = mid
        except Exception:
            break
        time.sleep(0.1)

    return lo


def fetch_ct_window(anchor_index, anchor_domain, anchor_ts, tree_size,
                    ct_log_url, window_size, out_file) -> int:
    """Fetch certificates before and after anchor index."""
    start = max(0, anchor_index - window_size)
    end = min(tree_size - 1, anchor_index + window_size)
    written = 0

    try:
        resp = requests.get(
            f"{ct_log_url}/get-entries",
            params={"start": start, "end": end},
            timeout=30,
        )
        resp.raise_for_status()
        entries = resp.json().get("entries", [])
    except requests.RequestException as exc:
        log.warning("Failed to fetch window at %d: %s", anchor_index, exc)
        return 0

    for i, entry in enumerate(entries):
        abs_index = start + i
        record = parse_ct_entry(entry, abs_index)
        if not record:
            continue

        if abs_index < anchor_index:
            position = "before"
        elif abs_index == anchor_index:
            position = "anchor"
        else:
            position = "after"

        record.update({
            "data_source": "ct_historical_window",
            "anchor_domain": anchor_domain,
            "anchor_ts": anchor_ts,
            "anchor_index": anchor_index,
            "window_position": position,
        })
        out_file.write(json.dumps(record, default=str) + "\n")
        written += 1

    return written


# ══════════════════════════════════════════════════════════════════════════════
# CLI ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════════════

def create_parser():
    """Create argument parser with subcommands for each mode."""
    parser = argparse.ArgumentParser(
        description="Unified SSL certificate and phishing label ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="mode", required=True, help="Ingestion mode")

    # ── live (integrated pipeline) ────────────────────────────────────────────
    live_pipeline = subparsers.add_parser(
        "live",
        help="Integrated pipeline: collect certs + refresh labels"
    )
    live_pipeline.add_argument("--certs", default="data/raw/certs_fallback.jsonl",
                               help="Certs output JSONL (default: certs_fallback.jsonl)")
    live_pipeline.add_argument("--labels", default="data/raw/phishtank_labels.jsonl",
                               help="Labels output JSONL (default: phishtank_labels.jsonl)")
    live_pipeline.add_argument("--duration", type=int, default=900,
                               help="Seconds per cert collection batch (default: 900 or 15 minutes)")
    live_pipeline.add_argument("--iterations", type=int, default=5,
                               help="Number of collection cycles (default: 5)")
    live_pipeline.add_argument("--ct-log-url",
                               default="https://ct.googleapis.com/logs/us1/argon2026h1/ct/v1",
                               help="CT log API URL")
    live_pipeline.add_argument("--batch-size", type=int, default=256,
                               help="CT entries per API call (default: 256)")
    live_pipeline.add_argument("--poll-delay", type=float, default=1.0,
                               help="Seconds between CT batches (default: 1.0)")
    live_pipeline.add_argument("--api-key",
                               help="PhishTank API key (or set PHISHTANK_API_KEY env var)")
    live_pipeline.add_argument("--phishtank-cache", default="phishtank_cache.csv",
                               help="PhishTank CSV cache file (default: phishtank_cache.csv)")
    live_pipeline.add_argument("--tranco-cache", default="tranco_top1m.csv",
                               help="Tranco CSV cache file (default: tranco_top1m.csv)")

    # ── live-certs ────────────────────────────────────────────────────────────
    live = subparsers.add_parser("live-certs", help="Poll CT log for live certificates")
    live.add_argument("-o", "--output", default="data/raw/certs_fallback.jsonl",
                      help="Output JSONL file (default: certs_fallback.jsonl)")
    live.add_argument("--ct-log-url", default="https://ct.googleapis.com/logs/us1/argon2026h1/ct/v1",
                      help="CT log API URL")
    live.add_argument("--batch-size", type=int, default=256,
                      help="Entries per API call (default: 256)")
    live.add_argument("--poll-delay", type=float, default=1.0,
                      help="Seconds between batches (default: 1.0)")
    live.add_argument("--start-index", type=int,
                      help="Starting CT log index (default: current tree size - batch_size)")
    live.add_argument("--duration", type=int, default=900,
                      help="Seconds per cert collection batch (default: 900 or 15 minutes)")

    # ── live-labels ────────────────────────────────────────────────────────────────
    labels = subparsers.add_parser(
        "live-labels",
        help="Fetch PhishTank labels for domains in cert file"
    )
    labels.add_argument("-c", "--certs", required=True,
                        help="Cert JSONL file to extract domains from (required)")
    labels.add_argument("-o", "--output", default="data/raw/phishtank_labels.jsonl",
                        help="Output JSONL file (default: data/raw/phishtank_labels.jsonl)")
    labels.add_argument("--api-key",
                        help="PhishTank API key (or set PHISHTANK_API_KEY env var)")
    labels.add_argument("--phishtank-cache", default="phishtank_cache.csv",
                        help="PhishTank CSV cache file (default: phishtank_cache.csv)")
    labels.add_argument("--tranco-cache", default="tranco_top1m.csv",
                        help="Tranco CSV cache file (default: tranco_top1m.csv)")

    # ── historical-phishing-labels ────────────────────────────────────────────────
    phishing_labels = subparsers.add_parser(
        "historical-phishing-labels",
        help="Import all PhishTank phishing domains as labels"
    )
    phishing_labels.add_argument("-o", "--output", default="data/raw/phishing_labels.jsonl",
                                 help="Output JSONL file (default: data/raw/phishing_labels.jsonl)")
    phishing_labels.add_argument("--api-key",
                                 help="PhishTank API key (or set PHISHTANK_API_KEY env var)")
    phishing_labels.add_argument("--phishtank-cache", default="phishtank_cache.csv",
                                 help="PhishTank CSV cache file (default: phishtank_cache.csv)")
    phishing_labels.add_argument("--overwrite", action="store_true",
                                 help="Overwrite existing file instead of appending (default: append)")

    # ── historical-phishing-certs ─────────────────────────────────────────────────
    hist = subparsers.add_parser("historical-phishing-certs", help="Fetch historical certs for phishing domains via crt.sh")
    hist.add_argument("-l", "--labels", required=True,
                      help="PhishTank labels JSONL file (required)")
    hist.add_argument("-o", "--output", default="data/raw/certs_phishtank_historical.jsonl",
                      help="Output JSONL file (default: certs_phishtank_historical.jsonl)")
    hist.add_argument("--max-domains", type=int,
                      help="Limit to first N domains (default: all)")
    hist.add_argument("--max-certs-per-domain", type=int, default=10,
                      help="Max certs to fetch per domain (default: 10)")
    hist.add_argument("--request-delay", type=float, default=1.5,
                      help="Seconds between crt.sh requests (default: 1.5)")

    # ── windows ───────────────────────────────────────────────────────────────
    windows = subparsers.add_parser("windows", help="Fetch CT windows around phishing certs")
    windows.add_argument("-p", "--phishing", required=True,
                         help="Phishing certs JSONL file (required)")
    windows.add_argument("-o", "--output", default="data/raw/certs_ct_historical_windows.jsonl",
                         help="Output JSONL file (default: certs_ct_historical_windows.jsonl)")
    windows.add_argument("--ct-log-url", default="https://ct.googleapis.com/logs/us1/argon2026h1/ct/v1",
                         help="CT log API URL")
    windows.add_argument("--window-size", type=int, default=500,
                         help="Certs to fetch before/after anchor (default: 500)")
    windows.add_argument("--max-anchors", type=int, default=100,
                         help="Max phishing certs to use as anchors (default: 100)")
    windows.add_argument("--request-delay", type=float, default=1.0,
                         help="Seconds between requests (default: 1.0)")

    return parser


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = create_parser()
    args = parser.parse_args()

    log.info("Starting ingestion mode: %s", args.mode)

    if args.mode == "live":
        run_live_pipeline(args)
    elif args.mode == "live-certs":
        run_live_certs(args)
    elif args.mode == "live-labels":
        run_live_labels(args)
    elif args.mode == "historical-phishing-labels":
        run_historical_phishing_labels(args)
    elif args.mode == "historical-phishing-certs":
        run_historical_phishing_certs(args)
    elif args.mode == "windows":
        run_windows(args)
    else:
        parser.print_help()
        sys.exit(1)

    log.info("Ingestion complete")


if __name__ == "__main__":
    main()
