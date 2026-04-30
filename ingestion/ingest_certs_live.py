#!/usr/bin/env python3
"""
CT Log direct poller — certstream-free sample collector.

Polls Google's Argon CT log directly via the RFC 6962 API.
Normalises entries to the same schema as certstream_ingest.py
so your feature engineering code works unchanged.

Install:
    pip install requests cryptography

Run:
    python ct_poller.py
"""

import base64
import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone

import requests
from cryptography import x509
from cryptography.hazmat.backends import default_backend

log = logging.getLogger("ct-poller")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# ── Config ────────────────────────────────────────────────────────────────────

CT_LOG_URL   = "https://ct.googleapis.com/logs/us1/argon2026h1/ct/v1"
OUTPUT_FILE  = "certs_fallback.jsonl"
BATCH_SIZE   = 256          # entries per API call (max 1000, keep low to be polite)
POLL_DELAY   = 1.0          # seconds between batches
START_INDEX  = None         # None = fetch current tree size and start from head

# ── Stats ─────────────────────────────────────────────────────────────────────

stats = {"fetched": 0, "parsed": 0, "errors": 0, "start_time": time.time()}

def print_stats():
    elapsed = time.time() - stats["start_time"]
    rate = stats["fetched"] / elapsed if elapsed else 0
    log.info("fetched:%d parsed:%d errors:%d rate:%.1f/s",
             stats["fetched"], stats["parsed"], stats["errors"], rate)

# ── CT log API ────────────────────────────────────────────────────────────────

def get_tree_size() -> int:
    resp = requests.get(f"{CT_LOG_URL}/get-sth", timeout=10)
    resp.raise_for_status()
    size = resp.json()["tree_size"]
    log.info("CT log tree size: %d", size)
    return size


def get_entries(start: int, end: int) -> list[dict]:
    resp = requests.get(
        f"{CT_LOG_URL}/get-entries",
        params={"start": start, "end": end},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("entries", [])


# ── Certificate parsing ───────────────────────────────────────────────────────

def parse_entry(entry: dict, index: int) -> dict | None:
    """
    Decode a raw CT log entry and extract the same fields as certstream_ingest.py.
    Returns None if the entry can't be parsed (e.g. pre-cert, malformed DER).
    """
    try:
        leaf_input = base64.b64decode(entry["leaf_input"])
        # CT MerkleTreeLeaf structure:
        # 2 bytes version + type, 8 bytes timestamp, 2 bytes entry type, then data
        # For X509LogEntry (type 0): 3-byte length prefix then DER cert
        # For PrecertLogEntry (type 1): skip (issuer key hash + TBS cert)
        entry_type = int.from_bytes(leaf_input[10:12], "big")
        if entry_type == 1:
            # precert — parse from extra_data instead
            extra = base64.b64decode(entry.get("extra_data", ""))
            # extra_data for precert: chain of certs, first is the precert itself
            # 3-byte length prefix for each cert
            if len(extra) < 3:
                return None
            cert_len = int.from_bytes(extra[:3], "big")
            der = extra[3:3 + cert_len]
        else:
            # x509 — cert follows 3-byte length prefix at offset 12
            cert_len = int.from_bytes(leaf_input[12:15], "big")
            der = leaf_input[15:15 + cert_len]

        cert = x509.load_der_x509_certificate(der, default_backend())

        # Extract SANs (all_domains equivalent)
        try:
            san_ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
            domains = san_ext.value.get_values_for_type(x509.DNSName)
        except x509.ExtensionNotFound:
            domains = []

        # Fall back to CN if no SANs
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
        issuer  = cert.issuer
        subject = cert.subject

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
            "subject": {
                "CN": name_attr(subject, oid.COMMON_NAME),
                "O":  name_attr(subject, oid.ORGANIZATION_NAME),
                "C":  name_attr(subject, oid.COUNTRY_NAME),
            },
            "issuer": {
                "CN": name_attr(issuer, oid.COMMON_NAME),
                "O":  name_attr(issuer, oid.ORGANIZATION_NAME),
                "C":  name_attr(issuer, oid.COUNTRY_NAME),
                "aggregated": issuer.rfc4514_string(),
            },
            "source": {
                "name": "Google 'Argon2026h1' log",
                "url":  CT_LOG_URL,
            },
        }

    except Exception as exc:
        stats["errors"] += 1
        log.debug("Parse error at index %d: %s", index, exc)
        return None


# ── Main loop ─────────────────────────────────────────────────────────────────

_out_file = None
_running  = True

def shutdown(sig, frame):
    global _running
    log.info("Shutting down …")
    _running = False
    print_stats()
    if _out_file:
        _out_file.close()
    sys.exit(0)


def main():
    global _out_file

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    _out_file = open(OUTPUT_FILE, "a", buffering=1)
    log.info("Writing to %s", OUTPUT_FILE)

    start = START_INDEX if START_INDEX is not None else get_tree_size() - BATCH_SIZE

    log.info("Starting poll from index %d (batch=%d)", start, BATCH_SIZE)

    while _running:
        try:
            end = start + BATCH_SIZE - 1
            entries = get_entries(start, end)

            if not entries:
                log.info("No new entries at index %d, waiting …", start)
                time.sleep(5)
                continue

            for i, entry in enumerate(entries):
                stats["fetched"] += 1
                record = parse_entry(entry, start + i)
                if record:
                    stats["parsed"] += 1
                    _out_file.write(json.dumps(record, default=str) + "\n")

            start += len(entries)

            if stats["fetched"] % 2_000 == 0:
                print_stats()

            time.sleep(POLL_DELAY)

        except requests.RequestException as exc:
            log.warning("API error: %s — retrying in 10s", exc)
            time.sleep(10)


if __name__ == "__main__":
    main()