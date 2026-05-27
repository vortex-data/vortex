
WITH RankedCustomers AS (
    SELECT 
        c.c_customer_sk,
        CONCAT(c.c_first_name, ' ', c.c_last_name) AS full_name,
        cd.cd_gender,
        cd.cd_marital_status,
        ca.ca_city,
        ca.ca_state,
        ROW_NUMBER() OVER (PARTITION BY ca.ca_state ORDER BY c.c_customer_sk) AS state_rank
    FROM customer c
    JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
    JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
    WHERE ca.ca_city IS NOT NULL
),
FilteredCustomers AS (
    SELECT 
        full_name,
        cd_gender,
        cd_marital_status,
        ca_city,
        ca_state
    FROM RankedCustomers
    WHERE state_rank <= 10
),
CustomerStats AS (
    SELECT 
        ca_state,
        COUNT(*) AS total_customers,
        STRING_AGG(full_name, ', ') AS customer_names,
        COUNT(CASE WHEN cd_gender = 'F' THEN 1 END) AS female_count,
        COUNT(CASE WHEN cd_marital_status = 'M' THEN 1 END) AS married_count
    FROM FilteredCustomers
    GROUP BY ca_state
)
SELECT 
    cs.ca_state,
    cs.total_customers,
    cs.customer_names,
    cs.female_count,
    cs.married_count,
    CASE 
        WHEN cs.total_customers > 50 THEN 'Large'
        WHEN cs.total_customers BETWEEN 20 AND 50 THEN 'Medium'
        ELSE 'Small'
    END AS customer_segment
FROM CustomerStats cs
ORDER BY cs.total_customers DESC;
