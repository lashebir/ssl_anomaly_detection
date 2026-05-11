{{
    config(
        materialized='table'
    )
}}

-- Explode multi-domain certificates to one row per domain
-- This is the base for domain-level feature extraction

WITH exploded AS (
    SELECT
        -- Certificate identifiers
        fingerprint,
        serial,

        -- Explode domains array (one row per domain)
        unnest(domains) as domain_raw,

        -- Timestamps
        timestamp,
        not_before,
        not_after,

        -- Issuer
        issuer_cn,
        issuer_org,
        issuer_country,

        -- Subject
        subject_cn,
        subject_org,
        subject_country,

        -- Certificate-level properties (computed before explosion)
        CASE
            WHEN array_to_string(domains, ',') LIKE '%*.%'
            THEN 1
            ELSE 0
        END as is_wildcard,

        array_length(domains) as san_count,

        CASE
            WHEN subject_cn = issuer_cn AND subject_org = issuer_org
            THEN 1
            ELSE 0
        END as is_self_signed,

        CASE WHEN subject_org IS NOT NULL THEN 1 ELSE 0 END as has_subject_org,

        -- Source metadata
        source,
        data_source

    FROM {{ ref('stg_certs') }}
)

SELECT
    -- Certificate identifiers
    fingerprint,
    serial,

    -- Normalize domain: lowercase, trim, remove leading wildcard
    lower(trim(regexp_replace(domain_raw, '^\*\.', ''))) as domain,

    -- Timestamps
    timestamp,
    not_before,
    not_after,

    -- Issuer
    issuer_cn,
    issuer_org,
    issuer_country,

    -- Subject
    subject_cn,
    subject_org,
    subject_country,

    -- Certificate-level features
    is_wildcard,
    san_count,
    is_self_signed,
    has_subject_org,

    -- Certificate age (days from issuance to now)
    CAST((epoch(current_timestamp) - not_before) / 86400.0 AS DOUBLE) as cert_age_days,

    -- Source metadata
    source,
    data_source

FROM exploded

-- Filter out empty domains after normalization
WHERE length(trim(regexp_replace(domain_raw, '^\*\.', ''))) > 0
