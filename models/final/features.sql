{{
    config(
        materialized='table'
    )
}}

-- Final features model - combines cert features, domain features, and labels
-- This is the output consumed by dashboard.py

WITH cert_domain AS (
    SELECT
        ce.fingerprint,
        ce.serial,
        ce.domain,
        ce.timestamp,
        ce.not_before,
        ce.not_after,

        -- Issuer
        ce.issuer_cn,
        ce.issuer_org,
        ce.issuer_country,

        -- Subject
        ce.subject_cn,
        ce.subject_org,
        ce.subject_country,

        -- Certificate-level features
        ce.is_wildcard,
        ce.san_count,
        ce.is_self_signed,
        ce.has_subject_org,
        ce.cert_age_days,

        -- Source metadata
        ce.source,
        ce.data_source

    FROM {{ ref('int_certs_exploded') }} ce
),

features_joined AS (
    SELECT
        cd.fingerprint,
        cd.serial,
        cd.domain,
        cd.timestamp,
        cd.not_before,
        cd.not_after,

        -- Issuer
        cd.issuer_org,

        -- Domain features (from int_domain_features)
        df.domain_length,
        df.entropy,
        df.subdomain_count,
        df.hyphen_count,
        df.digit_count,
        df.digit_ratio,
        df.vowel_consonant_ratio,
        df.consecutive_consonants,
        df.has_at_symbol,
        df.has_ip_address,
        df.tld,
        df.tld_risk,
        df.brand_distance,
        df.closest_brand,
        df.is_brand_lookalike,
        df.keyword_count,
        df.has_keyword,

        -- Certificate features
        cd.cert_age_days,
        cd.is_wildcard,
        cd.san_count,
        cd.is_self_signed,
        cd.has_subject_org,

        -- Labels (join with stg_labels)
        COALESCE(lbl.y, 0) as y,
        COALESCE(lbl.label_source, 'unknown') as label_source,
        lbl.label_timestamp,

        -- Source metadata
        cd.source,
        cd.data_source

    FROM cert_domain cd
    INNER JOIN {{ ref('int_domain_features') }} df
        ON (cd.fingerprint = df.fingerprint OR (cd.fingerprint IS NULL AND cd.serial = df.serial))
        AND cd.domain = df.domain
    LEFT JOIN {{ ref('stg_labels') }} lbl
        ON cd.domain = lbl.domain
)

SELECT * FROM features_joined

-- Optional: Filter to only labeled data for training
-- Uncomment to exclude unlabeled domains:
-- WHERE label_source IN ('phishtank', 'tranco')
