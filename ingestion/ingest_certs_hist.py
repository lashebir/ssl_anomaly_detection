#!/usr/bin/env python3
"""
crt.sh phishing cert fetcher.

Takes known phishing domains from phishtank_labels.jsonl, queries crt.sh
for their certificate history, and writes to certs_phishtank_historical.jsonl.

This file is kept SEPARATE from certs_ct.jsonl (your live CT log sample)
to preserve time series integrity. See DATASET STRATEGY at the bottom.

Install:
    pip install requests pandas

Run:
    python crtsh_fetcher.py
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# ── Config ────────────────────────────────────────────────────────────────────

LABELS_FILE     = "phishtank_labels.jsonl"
OUTPUT_FILE     = "certs_phishtank_historical.jsonl"
REQUEST_DELAY   = 1.5      # seconds between crt.sh requests — be polite
MAX_DOMAINS     = 2000     # cap to avoid multi-hour runs; None = all
MAX_CERTS_PER_DOMAIN = 10  # crt.sh returns many dupes; cap per domain

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("crtsh-fetcher")

# ── crt.sh query ──────────────────────────────────────────────────────────────

def fetch_certs_for_domain(domain: str) -> list[dict]:
    """
    Query crt.sh for all certificates ever issued for a domain.
    Returns a list of raw crt.sh records, capped at MAX_CERTS_PER_DOMAIN.
    Returns [] on any error (rate limit, timeout, bad JSON).
    """
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
        return resp.json()[:MAX_CERTS_PER_DOMAIN]
    except requests.RequestException as exc:
        log.warning("Request failed for %s: %s", domain, exc)
        return []
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning("Bad JSON from crt.sh for %s: %s", domain, exc)
        return []


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalise(raw: dict, phishing_domain: str, label_ts: str) -> dict | None:
    """
    Normalise a crt.sh record to the same schema as certs_ct.jsonl so
    your feature engineering code works on both files without branching.

    Extra fields added:
        data_source     — "crtsh_historical" (never "ct_live")
        phishing_domain — the PhishTank domain that triggered this lookup
        label_source    — "phishtank"
        label_ts        — timestamp of the PhishTank snapshot used
        y               — 1 (always — this file only contains phishing certs)

    crt.sh field reference:
        id              — crt.sh internal id (not the CT log index)
        issuer_name     — full issuer DN string
        common_name     — cert CN
        name_value      — SANs, newline-separated
        not_before      — "YYYY-MM-DD HH:MM:SS" string
        not_after       — "YYYY-MM-DD HH:MM:SS" string
        serial_number   — hex string
    """
    try:
        # Parse SANs from newline-separated name_value field
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

        def parse_ts(s):
            """crt.sh uses 'YYYY-MM-DD HH:MM:SS' UTC strings."""
            if not s:
                return None
            try:
                return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=timezone.utc).timestamp()
            except ValueError:
                return None

        # Parse issuer fields from the DN string
        # e.g. "C=US, O=Let's Encrypt, CN=R13"
        issuer_str = raw.get("issuer_name", "") or ""
        issuer_parts = dict(
            part.strip().split("=", 1)
            for part in issuer_str.split(",")
            if "=" in part
        )

        return {
            # ── Core fields (same as certs_ct.jsonl) ──────────────────────
            "schema_version": 1,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "cert_index": None,             # no CT index from crt.sh
            "fingerprint": None,            # not returned by crt.sh JSON API
            "serial":     raw.get("serial_number"),
            "not_before": parse_ts(raw.get("not_before")),
            "not_after":  parse_ts(raw.get("not_after")),
            "domains":    domains,
            "subject": {
                "CN": raw.get("common_name"),
                "O":  None,                 # not in crt.sh response
            },
            "issuer": {
                "CN":         issuer_parts.get("CN"),
                "O":          issuer_parts.get("O"),
                "C":          issuer_parts.get("C"),
                "aggregated": issuer_str,
            },
            "source": {
                "name": "crt.sh historical lookup",
                "url":  f"https://crt.sh/?id={raw.get('id', '')}",
            },
            # ── Provenance fields (not in certs_ct.jsonl) ─────────────────
            # These let you filter this file out of time series analysis
            # and include it only for classifier training.
            "data_source":     "crtsh_historical",
            "phishing_domain": phishing_domain,
            "label_source":    "phishtank",
            "label_ts":        label_ts,
            "y":               1,
        }
    except Exception as exc:
        log.debug("Normalise failed: %s", exc)
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not Path(LABELS_FILE).exists():
        log.error("%s not found — run phishtank.py first", LABELS_FILE)
        return

    labels = pd.read_json(LABELS_FILE, lines=True)
    domains = labels["domain"].dropna().unique().tolist()

    if MAX_DOMAINS:
        domains = domains[:MAX_DOMAINS]

    log.info("Fetching crt.sh certs for %d phishing domains → %s", len(domains), OUTPUT_FILE)

    stats = {"domains_queried": 0, "certs_written": 0, "domains_empty": 0, "errors": 0}
    label_ts = labels["label_ts"].iloc[0] if "label_ts" in labels.columns else \
               datetime.now(timezone.utc).isoformat()

    # Track already-written serials to deduplicate across domains
    seen_serials: set[str] = set()

    with open(OUTPUT_FILE, "w", buffering=1) as out:
        for i, domain in enumerate(domains):
            raw_certs = fetch_certs_for_domain(domain)
            stats["domains_queried"] += 1

            if not raw_certs:
                stats["domains_empty"] += 1
            else:
                for raw in raw_certs:
                    serial = raw.get("serial_number", "")
                    if serial and serial in seen_serials:
                        continue              # skip cross-domain dupes
                    record = normalise(raw, domain, label_ts)
                    if record:
                        out.write(json.dumps(record, default=str) + "\n")
                        stats["certs_written"] += 1
                        if serial:
                            seen_serials.add(serial)

            # Progress log every 50 domains
            if (i + 1) % 50 == 0:
                log.info(
                    "Progress: %d/%d domains | %d certs written | %d empty",
                    i + 1, len(domains),
                    stats["certs_written"], stats["domains_empty"],
                )

            time.sleep(REQUEST_DELAY)

    log.info("Done — %s", stats)
    print(f"\nWrote {stats['certs_written']} certs to {OUTPUT_FILE}")
    print(f"Domains queried: {stats['domains_queried']} | empty: {stats['domains_empty']}")


if __name__ == "__main__":
    main()


# ── DATASET STRATEGY ──────────────────────────────────────────────────────────
#
# You now have three files with distinct roles:
#
#   certs_ct.jsonl
#       Source: live Google CT log poll
#       data_source field: absent (or "ct_live" if you add it)
#       Labels: mostly y=0 (unknown), joined from phishtank_labels.jsonl
#       Use for: time series analysis, campaign detection, production simulation
#       DO NOT mix historical certs into this file.
#
#   certs_phishtank_historical.jsonl   (this file)
#       Source: crt.sh lookups for PhishTank domains
#       data_source field: "crtsh_historical"
#       Labels: y=1 (all phishing, by construction)
#       Use for: classifier training/eval ONLY
#       NOT for time series — timestamps are historical and non-representative.
#
#   phishtank_labels.jsonl
#       Source: PhishTank verified+online feed
#       Use for: joining labels onto certs_ct.jsonl in the notebook
#
#
# Notebook join pattern:
#
#   # For classifier training — combine both cert files, filter to labeled rows
#   df_live    = pd.read_json("certs_ct.jsonl", lines=True)
#   df_hist    = pd.read_json("certs_phishtank_historical.jsonl", lines=True)
#   labels     = pd.read_json("phishtank_labels.jsonl", lines=True)
#
#   # Label the live data
#   df_live_domains = df_live.explode("domains").rename(columns={"domains":"domain"})
#   df_live_domains["domain"] = df_live_domains["domain"].str.lower().str.lstrip("*.")
#   df_live_labeled = df_live_domains.merge(labels[["domain","y"]], on="domain", how="left")
#   df_live_labeled["y"] = df_live_labeled["y"].fillna(0).astype(int)
#
#   # Historical data is pre-labeled (y=1 already in the file)
#   df_hist_domains = df_hist.explode("domains").rename(columns={"domains":"domain"})
#
#   # Combine for classifier training
#   df_classifier = pd.concat([df_live_labeled, df_hist_domains], ignore_index=True)
#   df_classifier["data_source"] = df_classifier["data_source"].fillna("ct_live")
#
#   # For time series only — never include historical
#   df_timeseries = df_live_labeled[df_live_labeled["data_source"] != "crtsh_historical"]
