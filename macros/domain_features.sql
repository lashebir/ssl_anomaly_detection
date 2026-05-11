-- Macros for domain-level feature extraction

{% macro shannon_entropy(text_col) %}
/*
Calculate Shannon entropy of a text string (excluding dots).
Higher entropy = more random = suspicious.

Formula: H(X) = -Σ(p(x) * log2(p(x)))
*/
(
    WITH char_counts AS (
        SELECT
            char,
            count(*) as cnt,
            sum(count(*)) OVER () as total
        FROM (
            SELECT unnest(string_split(regexp_replace({{ text_col }}, '\.', ''), '')) as char
        )
        GROUP BY char
    )
    SELECT COALESCE(-sum((cnt::DOUBLE / total::DOUBLE) * log2(cnt::DOUBLE / total::DOUBLE)), 0.0)
    FROM char_counts
)
{% endmacro %}


{% macro extract_tld(domain_col) %}
/*
Extract top-level domain from a domain name.
Examples:
  example.com -> com
  example.co.uk -> uk
  mail.google.com -> com
*/
regexp_extract({{ domain_col }}, '\.([^.]+)$', 1)
{% endmacro %}


{% macro extract_subdomain(domain_col) %}
/*
Extract subdomain portion (everything before registrable domain).
Returns NULL if no subdomain.

Examples:
  www.example.com -> www
  mail.secure.example.com -> mail.secure
  example.com -> NULL
*/
CASE
    WHEN length({{ domain_col }}) - length(replace({{ domain_col }}, '.', '')) > 1
    THEN regexp_extract({{ domain_col }}, '^(.+)\.([^.]+\.[^.]+)$', 1)
    ELSE NULL
END
{% endmacro %}


{% macro count_subdomains(domain_col) %}
/*
Count number of subdomain levels.
Examples:
  www.example.com -> 1
  mail.secure.example.com -> 2
  example.com -> 0
*/
CASE
    WHEN {{ extract_subdomain(domain_col) }} IS NULL THEN 0
    ELSE length({{ extract_subdomain(domain_col) }}) - length(replace({{ extract_subdomain(domain_col) }}, '.', '')) + 1
END
{% endmacro %}


{% macro count_consecutive_consonants(domain_col) %}
/*
Count maximum consecutive consonants in domain.
Legitimate domains rarely have >4 consecutive consonants.
*/
(
    WITH chars AS (
        SELECT
            char,
            CASE
                WHEN char ~ '[a-z]' AND char NOT IN ('a', 'e', 'i', 'o', 'u')
                THEN 1
                ELSE 0
            END as is_consonant,
            row_number() OVER (ORDER BY pos) -
                sum(CASE WHEN char ~ '[a-z]' AND char NOT IN ('a', 'e', 'i', 'o', 'u') THEN 1 ELSE 0 END)
                OVER (ORDER BY pos) as grp
        FROM (
            SELECT unnest(string_split(lower(regexp_replace({{ domain_col }}, '\.', '')), '')) as char,
                   generate_series(1, length(regexp_replace({{ domain_col }}, '\.', ''))) as pos
        )
    ),
    runs AS (
        SELECT count(*) as run_length
        FROM chars
        WHERE is_consonant = 1
        GROUP BY grp
    )
    SELECT COALESCE(max(run_length), 0)
    FROM runs
)
{% endmacro %}


{% macro vowel_consonant_ratio(domain_col) %}
/*
Ratio of vowels to consonants (excluding dots and digits).
Suspicious domains may have unusual ratios.
*/
(
    WITH letters AS (
        SELECT
            sum(CASE WHEN char IN ('a', 'e', 'i', 'o', 'u') THEN 1 ELSE 0 END) as vowels,
            sum(CASE WHEN char NOT IN ('a', 'e', 'i', 'o', 'u') THEN 1 ELSE 0 END) as consonants
        FROM (
            SELECT unnest(string_split(lower(regexp_replace({{ domain_col }}, '[^a-z]', '', 'g')), '')) as char
        )
    )
    SELECT
        CASE
            WHEN consonants > 0 THEN vowels::DOUBLE / consonants::DOUBLE
            ELSE 0.0
        END
    FROM letters
)
{% endmacro %}


{% macro has_ip_pattern(domain_col) %}
/*
Check if domain contains IP address pattern.
Phishing often uses IP addresses instead of domains.
*/
CASE
    WHEN {{ domain_col }} ~ '\d{1,3}[-\.]\d{1,3}[-\.]\d{1,3}[-\.]\d{1,3}'
    THEN 1
    ELSE 0
END
{% endmacro %}
