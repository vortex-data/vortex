
SELECT 
    ca.ca_city, 
    ca.ca_state, 
    COUNT(DISTINCT c.c_customer_id) AS customer_count, 
    AVG(cd.cd_purchase_estimate) AS avg_purchase_estimate,
    SUM(CASE WHEN cd.cd_gender = 'F' THEN 1 ELSE 0 END) AS female_customers,
    SUM(CASE WHEN cd.cd_gender = 'M' THEN 1 ELSE 0 END) AS male_customers,
    STRING_AGG(DISTINCT cd.cd_marital_status, ', ') AS marital_statuses,
    STRING_AGG(DISTINCT cd.cd_education_status, ', ') AS education_levels
FROM 
    customer_address ca
JOIN 
    customer c ON ca.ca_address_sk = c.c_current_addr_sk
JOIN 
    customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
WHERE 
    ca.ca_state IN ('CA', 'NY')
GROUP BY 
    ca.ca_city, 
    ca.ca_state
ORDER BY 
    customer_count DESC
LIMIT 50;
