{{
    config(
        materialized='view'
    )
}}

-- Staging model for domain labels
-- Loads from DuckDB raw.labels table (populated from JSONL)

SELECT
    -- Domain (normalized to lowercase, trimmed)
    lower(trim(domain)) as domain,

    -- Label (0 = legitimate, 1 = phishing)
    y,

    -- Label source (phishtank, tranco, unknown)
    label_source,

    -- Timestamp when label was fetched
    label_ts as label_timestamp

FROM {{ source('raw', 'labels') }}

-- Filter out invalid domains
WHERE domain IS NOT NULL
  AND length(trim(domain)) > 0
  AND y IS NOT NULL
  AND label_source IS NOT NULL
