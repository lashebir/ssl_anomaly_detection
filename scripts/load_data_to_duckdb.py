#!/usr/bin/env python3
"""
Load streaming certificate and label data into DuckDB.

This script:
1. Creates a DuckDB database (feature_store.duckdb)
2. Loads JSONL files into raw tables
3. Prepares data for dbt transformation

Usage:
    python scripts/load_data_to_duckdb.py
    python scripts/load_data_to_duckdb.py --certs sources/raw/certs_streaming_48h.jsonl --labels sources/raw/labels_streaming_48h.jsonl
"""

import argparse
import sys
from pathlib import Path

try:
    import duckdb
except ImportError:
    print("ERROR: duckdb not installed")
    print("Install: pip install duckdb")
    sys.exit(1)


def load_jsonl_to_duckdb(
    db_path: str,
    certs_file: str,
    labels_file: str,
    verbose: bool = True
):
    """
    Load JSONL files into DuckDB.

    Args:
        db_path: Path to DuckDB database file
        certs_file: Path to certificates JSONL file
        labels_file: Path to labels JSONL file
        verbose: Print progress messages
    """
    certs_path = Path(certs_file)
    labels_path = Path(labels_file)

    # Check input files
    if not certs_path.exists():
        print(f"ERROR: Certs file not found: {certs_file}")
        print(f"\nExpected file from streaming ingestion.")
        print(f"Run: ./scripts/stream_48_hours_bg.sh")
        sys.exit(1)

    if not labels_path.exists():
        print(f"ERROR: Labels file not found: {labels_file}")
        sys.exit(1)

    if verbose:
        print("═" * 80)
        print("  Loading Data into DuckDB Feature Store")
        print("═" * 80)
        print(f"\nDatabase: {db_path}")
        print(f"Certs:    {certs_file}")
        print(f"Labels:   {labels_file}")
        print()

    # Connect to DuckDB
    con = duckdb.connect(db_path)

    # Install and load extensions
    if verbose:
        print("Installing DuckDB extensions...")
    con.execute("INSTALL json;")
    con.execute("INSTALL parquet;")
    con.execute("LOAD json;")
    con.execute("LOAD parquet;")

    # Create raw schema
    if verbose:
        print("Creating raw schema...")
    con.execute("CREATE SCHEMA IF NOT EXISTS raw;")

    # Drop existing tables
    con.execute("DROP TABLE IF EXISTS raw.certs;")
    con.execute("DROP TABLE IF EXISTS raw.labels;")

    # Load certificates from JSONL
    if verbose:
        print(f"\nLoading certificates from {certs_file}...")
    con.execute(f"""
        CREATE TABLE raw.certs AS
        SELECT * FROM read_json_auto(
            '{certs_file}',
            format='newline_delimited',
            maximum_object_size=10485760
        );
    """)

    cert_count = con.execute("SELECT count(*) FROM raw.certs").fetchone()[0]
    if verbose:
        print(f"  ✅ Loaded {cert_count:,} certificates")

    # Load labels from JSONL
    if verbose:
        print(f"\nLoading labels from {labels_file}...")
    con.execute(f"""
        CREATE TABLE raw.labels AS
        SELECT * FROM read_json_auto(
            '{labels_file}',
            format='newline_delimited',
            maximum_object_size=10485760
        );
    """)

    label_count = con.execute("SELECT count(*) FROM raw.labels").fetchone()[0]
    if verbose:
        print(f"  ✅ Loaded {label_count:,} labels")

    # Show label distribution
    if verbose:
        print("\n  Label Distribution:")
        distribution = con.execute("""
            SELECT
                label_source,
                count(*) as count,
                round(100.0 * count(*) / sum(count(*)) over (), 2) as pct
            FROM raw.labels
            GROUP BY label_source
            ORDER BY count DESC
        """).fetchall()

        for source, count, pct in distribution:
            print(f"    {source:12s}: {count:7,} ({pct:5.2f}%)")

    # Show sample domains
    if verbose:
        print("\n  Sample domains (first 5):")
        samples = con.execute("""
            SELECT DISTINCT unnest(domains) as domain
            FROM raw.certs
            LIMIT 5
        """).fetchall()

        for (domain,) in samples:
            print(f"    - {domain}")

    con.close()

    if verbose:
        print("\n" + "═" * 80)
        print("  ✅ Data Loading Complete!")
        print("═" * 80)
        print("\n  Next step: Run dbt models to generate features")
        print("    dbt run")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Load streaming CT log data into DuckDB feature store"
    )
    parser.add_argument(
        "--db",
        default="feature_store.duckdb",
        help="Path to DuckDB database (default: feature_store.duckdb)"
    )
    parser.add_argument(
        "--certs",
        default="sources/raw/certs_streaming_48h.jsonl",
        help="Path to certificates JSONL file"
    )
    parser.add_argument(
        "--labels",
        default="sources/raw/labels_streaming_48h.jsonl",
        help="Path to labels JSONL file"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )

    args = parser.parse_args()

    load_jsonl_to_duckdb(
        db_path=args.db,
        certs_file=args.certs,
        labels_file=args.labels,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
