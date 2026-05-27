WITH AddressInfo AS (
    SELECT 
        ca_state,
        ca_city,
        UPPER(ca_street_name) AS upper_street_name,
        LOWER(ca_city) AS lower_city,
        CONCAT_WS(',', ca_street_number, ca_street_name, ca_city, ca_state) AS full_address,
        LENGTH(ca_street_name) AS street_name_length,
        LENGTH(ca_city) AS city_length
    FROM 
        customer_address
),
DemographicInfo AS (
    SELECT 
        cd_gender,
        cd_marital_status,
        cd_education_status,
        COUNT(cd_demo_sk) AS demographic_count
    FROM 
        customer_demographics
    WHERE 
        cd_purchase_estimate > 100
    GROUP BY 
        cd_gender,
        cd_marital_status,
        cd_education_status
),
AggregateInfo AS (
    SELECT 
        ai.ca_state,
        di.cd_gender,
        di.cd_marital_status,
        di.cd_education_status,
        AVG(ai.street_name_length) AS avg_street_name_length,
        SUM(di.demographic_count) AS total_demographics
    FROM 
        AddressInfo ai
    JOIN 
        DemographicInfo di ON ai.ca_city = di.cd_gender  
    GROUP BY 
        ai.ca_state, 
        di.cd_gender, 
        di.cd_marital_status, 
        di.cd_education_status
)
SELECT 
    a.ca_state,
    d.cd_gender,
    d.cd_marital_status,
    d.cd_education_status,
    a.avg_street_name_length,
    a.total_demographics
FROM 
    AggregateInfo a
JOIN 
    DemographicInfo d ON a.ca_state = d.cd_gender  
WHERE 
    a.total_demographics > 10
ORDER BY 
    a.avg_street_name_length DESC, 
    a.total_demographics ASC;