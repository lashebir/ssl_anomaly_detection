{{
    config(
        materialized='table'
    )
}}

-- Domain-level feature extraction
-- Computes features for each domain in the certificate dataset

WITH base_features AS (
    SELECT
        fingerprint,
        serial,
        domain,

        -- Basic domain features
        length(domain) as domain_length,
        {{ shannon_entropy('domain') }} as entropy,
        {{ count_subdomains('domain') }} as subdomain_count,
        length(domain) - length(replace(domain, '-', '')) as hyphen_count,
        length(regexp_replace(domain, '[^0-9]', '', 'g')) as digit_count,
        length(regexp_replace(domain, '[^0-9]', '', 'g'))::DOUBLE / NULLIF(length(regexp_replace(domain, '\.', '', 'g')), 0)::DOUBLE as digit_ratio,

        -- Advanced domain features
        {{ vowel_consonant_ratio('domain') }} as vowel_consonant_ratio,
        {{ count_consecutive_consonants('domain') }} as consecutive_consonants,
        CASE WHEN domain LIKE '%@%' THEN 1 ELSE 0 END as has_at_symbol,
        {{ has_ip_pattern('domain') }} as has_ip_address,

        -- TLD extraction
        {{ extract_tld('domain') }} as tld

    FROM {{ ref('int_certs_exploded') }}
),

tld_features AS (
    SELECT
        bf.*,

        -- TLD risk score (join with seed data)
        COALESCE(tld_risk.risk_score, 1) as tld_risk

    FROM base_features bf
    LEFT JOIN {{ ref('tld_risk_scores') }} tld_risk
        ON bf.tld = tld_risk.tld
),

brand_features AS (
    SELECT
        tf.*,

        -- Brand distance: minimum Levenshtein distance to any brand
        (
            SELECT min(levenshtein(
                regexp_extract(tf.domain, '^([^.]+)', 1),  -- registrable domain
                brand
            ))
            FROM {{ ref('brands') }}
        ) as brand_distance,

        -- Closest brand
        (
            SELECT brand
            FROM {{ ref('brands') }}
            ORDER BY levenshtein(
                regexp_extract(tf.domain, '^([^.]+)', 1),
                brand
            )
            LIMIT 1
        ) as closest_brand

    FROM tld_features tf
),

keyword_features AS (
    SELECT
        bf.*,

        -- Keyword count: how many phishing keywords in domain
        (
            SELECT count(*)
            FROM {{ ref('phishing_keywords') }}
            WHERE position(keyword IN lower(bf.domain)) > 0
        ) as keyword_count

    FROM brand_features bf
)

SELECT
    fingerprint,
    serial,
    domain,

    -- Basic domain features
    domain_length,
    entropy,
    subdomain_count,
    hyphen_count,
    digit_count,
    digit_ratio,

    -- Advanced domain features
    vowel_consonant_ratio,
    consecutive_consonants,
    has_at_symbol,
    has_ip_address,

    -- TLD features
    tld,
    tld_risk,

    -- Brand/typosquatting features
    brand_distance,
    closest_brand,
    CASE WHEN brand_distance <= 3 THEN 1 ELSE 0 END as is_brand_lookalike,

    -- Keyword features
    keyword_count,
    CASE WHEN keyword_count > 0 THEN 1 ELSE 0 END as has_keyword

FROM keyword_features
