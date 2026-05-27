
WITH Address_summary AS (
    SELECT 
        ca_state,
        COUNT(DISTINCT ca_address_id) AS unique_addresses,
        SUM(CASE WHEN LENGTH(ca_street_name) > 20 THEN 1 ELSE 0 END) AS long_street_names,
        ARRAY_AGG(DISTINCT ca_city || ', ' || ca_street_name) AS city_street_combinations
    FROM 
        customer_address 
    GROUP BY 
        ca_state
),
Demographics_summary AS (
    SELECT 
        cd_gender,
        COUNT(*) AS num_customers,
        AVG(cd_purchase_estimate) AS avg_purchase_estimate,
        STRING_AGG(DISTINCT cd_education_status, ', ') AS education_levels
    FROM 
        customer_demographics 
    GROUP BY 
        cd_gender
),
Combined_summary AS (
    SELECT 
        a.ca_state,
        a.unique_addresses,
        a.long_street_names,
        a.city_street_combinations,
        d.cd_gender,
        d.num_customers,
        d.avg_purchase_estimate,
        d.education_levels
    FROM 
        Address_summary a 
    JOIN 
        Demographics_summary d 
    ON 
        a.unique_addresses > 10 AND d.num_customers > 50
)
SELECT 
    ca_state,
    unique_addresses,
    long_street_names,
    city_street_combinations,
    cd_gender,
    num_customers,
    avg_purchase_estimate,
    education_levels 
FROM 
    Combined_summary 
ORDER BY 
    unique_addresses DESC, num_customers DESC;
