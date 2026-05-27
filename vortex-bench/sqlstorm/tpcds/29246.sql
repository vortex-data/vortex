
WITH StringBenchmark AS (
    SELECT 
        c.c_customer_id,
        CONCAT(c.c_first_name, ' ', c.c_last_name) AS full_name,
        LENGTH(c.c_first_name) AS first_name_length,
        LENGTH(c.c_last_name) AS last_name_length,
        UPPER(c.c_first_name) AS upper_first_name,
        LOWER(c.c_last_name) AS lower_last_name,
        SUBSTRING(c.c_email_address, POSITION('@' IN c.c_email_address) + 1) AS email_domain,
        REPLACE(c.c_email_address, '.', '-') AS modified_email,
        REGEXP_REPLACE(c.c_email_address, '^[^@]+', 'user') AS anonymized_email,
        CASE 
            WHEN LENGTH(c.c_email_address) > 40 THEN 'Long Email' 
            ELSE 'Short Email' 
        END AS email_length_category,
        ROW_NUMBER() OVER (PARTITION BY c.c_customer_id ORDER BY c.c_customer_sk) AS customer_rank
    FROM 
        customer c
    WHERE 
        c.c_birth_year BETWEEN 1980 AND 2000
),
StringCounts AS (
    SELECT 
        full_name,
        COUNT(*) AS name_count,
        MAX(first_name_length) AS max_first_name_length,
        MIN(last_name_length) AS min_last_name_length,
        SUM(CASE WHEN email_length_category = 'Long Email' THEN 1 ELSE 0 END) AS long_email_count
    FROM 
        StringBenchmark
    GROUP BY 
        full_name
)
SELECT 
    sb.full_name,
    sb.first_name_length,
    sb.last_name_length,
    sc.name_count,
    sc.max_first_name_length,
    sc.min_last_name_length,
    sc.long_email_count,
    sb.upper_first_name,
    sb.lower_last_name,
    sb.email_domain,
    sb.modified_email,
    sb.anonymized_email,
    sb.customer_rank
FROM 
    StringBenchmark sb
JOIN 
    StringCounts sc ON sb.full_name = sc.full_name
ORDER BY 
    sb.first_name_length DESC, 
    sb.last_name_length ASC
LIMIT 100;
