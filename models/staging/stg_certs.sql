{{
    config(
        materialized='view'
    )
}}

-- Staging model for raw certificates
-- Loads from DuckDB raw.certs table (populated from JSONL)

SELECT
    -- Identifiers
    fingerprint,
    serial,

    -- Timestamps
    timestamp,
    not_before,
    not_after,

    -- Domains (JSON array)
    domains,

    -- Issuer fields (JSON object)
    issuer,
    issuer.CN as issuer_cn,
    issuer.O as issuer_org,
    issuer.C as issuer_country,

    -- Subject fields (JSON object)
    subject,
    subject.CN as subject_cn,
    subject.O as subject_org,
    subject.C as subject_country,

    -- Source metadata
    source,
    data_source

FROM {{ source('raw', 'certs') }}

-- Filter out any malformed records
-- Allow NULL fingerprint if serial exists (for historical data)
WHERE (fingerprint IS NOT NULL OR serial IS NOT NULL)
  AND domains IS NOT NULL
  AND array_length(domains) > 0
