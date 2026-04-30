#!/usr/bin/env python3
"""
CT Log historical window fetcher.

For each phishing cert in certs_phishtank_historical.jsonl, fetches a window
of CT log entries surrounding its issuance time. This gives you real temporal
context for time series analysis — what was being issued alongside each phishing
cert — without contaminating certs_ct.jsonl.

Output: certs_ct_historical_windows.jsonl
    Same schema as certs_ct.jsonl, with extra fields:
        anchor_domain   — the phishing domain that anchored this window
        anchor_ts       — issuance timestamp of the anchor cert
        window_position — position relative to anchor ("before", "anchor", "after")

Install:
    pip install requests cryptography pandas

Run:
    python ct_historical_windows.py
"""

import base64
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from cryptography import x509
from cryptography.hazmat.backends import default_backend

log = logging.getLogger("ct-windows")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# ── Config ────────────────────────────────────────────────────────────────────

CT_LOG_URL      = "https://ct.googleapis.com/logs/us1/argon2026h1/ct/v1"
PHISHING_FILE   = "certs_phishtank_historical.jsonl"
OUTPUT_FILE     = "certs_ct_historical_windows.jsonl"
WINDOW_SIZE     = 500     # fetch N certs before and after each anchor
MAX_ANCHORS     = 100     # how many phishing certs to build windows for
REQUEST_DELAY   = 1.0

# ── CT log helpers ────────────────────────────────────────────────────────────

def get_tree_size() -> int:
    r = requests.get(f"{CT_LOG_URL}/get-sth", timeout=10)
    r.raise_for_status()
    return r.json()["tree_size"]


def get_entries(start: int, end: int) -> list[dict]:
    r = requests.get(
        f"{CT_LOG_URL}/get-entries",
        params={"start": start, "end": end},
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("entries", [])


def timestamp_to_index(target_ts: float, tree_size: int) -> int:
    """
    Binary search the CT log to find the index whose issuance time is
    closest to target_ts (Unix timestamp).

    Strategy: fetch single entries at binary search midpoints, decode
    the timestamp from the MerkleTreeLeaf header (bytes 2-10, milliseconds).
    This avoids downloading full certs during the search.
    """
    lo, hi = 0, tree_size - 1

    for _ in range(30):     # 30 iterations → precision of ~1 in 2^30
        if lo >= hi:
            break
        mid = (lo + hi) // 2
        try:
            entries = get_entries(mid, mid)
            if not entries:
                break
            leaf = base64.b64decode(entries[0]["leaf_input"])
            # MerkleTreeLeaf: 1 byte version, 1 byte type, 8 bytes timestamp (ms)
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


# ── Certificate parser (same as ct_poller.py) ─────────────────────────────────

def parse_entry(entry: dict, index: int) -> dict | None:
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
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "cert_index": index,
            "fingerprint": cert.fingerprint(cert.signature_hash_algorithm).hex(":").upper()
                           if cert.signature_hash_algorithm else None,
            "serial":     str(cert.serial_number),
            "not_before": cert.not_valid_before_utc.timestamp(),
            "not_after":  cert.not_valid_after_utc.timestamp(),
            "domains":    list(domains),
            "subject": {"CN": attr(cert.subject, oid.COMMON_NAME),
                        "O":  attr(cert.subject, oid.ORGANIZATION_NAME)},
            "issuer":  {"CN": attr(cert.issuer,  oid.COMMON_NAME),
                        "O":  attr(cert.issuer,  oid.ORGANIZATION_NAME),
                        "aggregated": cert.issuer.rfc4514_string()},
            "source": {"name": "Google 'Argon2026h1' log", "url": CT_LOG_URL},
        }
    except Exception:
        return None


# ── Window fetcher ────────────────────────────────────────────────────────────

def fetch_window(
    anchor_index: int,
    anchor_domain: str,
    anchor_ts: float,
    tree_size: int,
    out_file,
) -> int:
    """
    Fetch WINDOW_SIZE certs before and after anchor_index.
    Tags each record with anchor metadata and window_position.
    Returns number of records written.
    """
    start = max(0, anchor_index - WINDOW_SIZE)
    end   = min(tree_size - 1, anchor_index + WINDOW_SIZE)
    written = 0

    try:
        entries = get_entries(start, end)
    except requests.RequestException as exc:
        log.warning("Failed to fetch window at %d: %s", anchor_index, exc)
        return 0

    for i, entry in enumerate(entries):
        abs_index = start + i
        record = parse_entry(entry, abs_index)
        if not record:
            continue

        if abs_index < anchor_index:
            position = "before"
        elif abs_index == anchor_index:
            position = "anchor"
        else:
            position = "after"

        record.update({
            "data_source":     "ct_historical_window",
            "anchor_domain":   anchor_domain,
            "anchor_ts":       anchor_ts,
            "anchor_index":    anchor_index,
            "window_position": position,
        })
        out_file.write(json.dumps(record, default=str) + "\n")
        written += 1

    return written


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not Path(PHISHING_FILE).exists():
        log.error("%s not found — run crtsh_fetcher.py first", PHISHING_FILE)
        return

    df_phish = pd.read_json(PHISHING_FILE, lines=True)

    # Pick anchors — prefer certs with valid not_before timestamps
    df_anchors = (df_phish
                  .dropna(subset=["not_before", "domains"])
                  .drop_duplicates(subset=["serial"])
                  .head(MAX_ANCHORS))

    log.info("Using %d phishing certs as time series anchors", len(df_anchors))

    tree_size = get_tree_size()
    log.info("CT log tree size: %d", tree_size)

    stats = {"anchors_processed": 0, "certs_written": 0}

    with open(OUTPUT_FILE, "w", buffering=1) as out:
        for _, row in df_anchors.iterrows():
            anchor_ts     = row["not_before"]
            anchor_domain = row["domains"][0] if isinstance(row["domains"], list) else row["domains"]

            log.info(
                "Finding CT index for %s (ts=%s) …",
                anchor_domain,
                datetime.fromtimestamp(anchor_ts, tz=timezone.utc).isoformat(),
            )

            anchor_index = timestamp_to_index(anchor_ts, tree_size)
            log.info("  → index %d — fetching ±%d window", anchor_index, WINDOW_SIZE)

            written = fetch_window(anchor_index, anchor_domain, anchor_ts, tree_size, out)
            stats["certs_written"] += written
            stats["anchors_processed"] += 1

            log.info("  → wrote %d records", written)
            time.sleep(REQUEST_DELAY)

    log.info("Done — %s", stats)
    print(f"\nWrote {stats['certs_written']:,} records to {OUTPUT_FILE}")
    print(f"Anchors processed: {stats['anchors_processed']}")
    print("\nLoad in notebook:")
    print("  df_windows = pd.read_json('certs_ct_historical_windows.jsonl', lines=True)")
    print("  # time series analysis: filter to before/after, group by anchor_domain")
    print("  # df_windows[df_windows['window_position'] == 'before']")


if __name__ == "__main__":
    main()
