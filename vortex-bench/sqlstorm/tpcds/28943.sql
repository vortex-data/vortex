
WITH CustomerInfo AS (
    SELECT 
        c.c_customer_sk,
        CONCAT(c.c_first_name, ' ', c.c_last_name) AS full_name,
        cd.cd_gender,
        cd.cd_marital_status,
        cd.cd_purchase_estimate,
        cd.cd_credit_rating,
        CONCAT(ca.ca_street_number, ' ', ca.ca_street_name, ' ', ca.ca_street_type) AS full_address,
        ca.ca_city,
        ca.ca_state,
        ca.ca_zip,
        ca.ca_country
    FROM 
        customer c
    JOIN 
        customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
    JOIN 
        customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
),
AggregateData AS (
    SELECT 
        COUNT(*) AS total_customers,
        COUNT(DISTINCT full_name) AS unique_customers,
        AVG(cd_purchase_estimate) AS avg_purchase_estimate,
        MAX(cd_purchase_estimate) AS max_purchase_estimate,
        MIN(cd_purchase_estimate) AS min_purchase_estimate,
        cd_gender
    FROM 
        CustomerInfo
    GROUP BY 
        cd_gender
),
DetailedAddress AS (
    SELECT 
        full_name,
        full_address
    FROM 
        CustomerInfo
    WHERE 
        ca_state = 'CA' 
    ORDER BY 
        ca_city
)
SELECT 
    a.total_customers,
    a.unique_customers,
    a.avg_purchase_estimate,
    a.max_purchase_estimate,
    a.min_purchase_estimate,
    a.cd_gender,
    d.full_name,
    d.full_address
FROM 
    AggregateData a
JOIN 
    DetailedAddress d ON 1 = 1
ORDER BY 
    a.cd_gender, d.full_name;
