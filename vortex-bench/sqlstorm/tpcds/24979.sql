
WITH ranked_customers AS (
    SELECT 
        c.c_customer_sk,
        c.c_first_name,
        c.c_last_name,
        cd.cd_gender,
        cd.cd_marital_status,
        ROW_NUMBER() OVER (PARTITION BY cd.cd_gender ORDER BY cd.cd_purchase_estimate DESC) AS purchase_rank
    FROM customer AS c
    INNER JOIN customer_demographics AS cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
    WHERE cd.cd_credit_rating IS NOT NULL
),
high_value_customers AS (
    SELECT 
        rc.c_customer_sk,
        rc.c_first_name,
        rc.c_last_name,
        rc.cd_gender,
        rc.cd_marital_status,
        COALESCE(SUM(ws.ws_net_paid), 0) AS total_spent,
        COUNT(ws.ws_order_number) AS purchase_count
    FROM ranked_customers AS rc
    LEFT JOIN web_sales AS ws ON rc.c_customer_sk = ws.ws_bill_customer_sk
    WHERE rc.purchase_rank <= 5
    GROUP BY rc.c_customer_sk, rc.c_first_name, rc.c_last_name, rc.cd_gender, rc.cd_marital_status
),
customer_location AS (
    SELECT 
        c.c_customer_sk,
        ca.ca_city,
        ca.ca_state,
        ROW_NUMBER() OVER (PARTITION BY c.c_customer_sk ORDER BY ca.ca_city) AS city_rank
    FROM customer AS c
    JOIN customer_address AS ca ON c.c_current_addr_sk = ca.ca_address_sk
),
customer_with_locations AS (
    SELECT 
        hvc.c_customer_sk,
        hvc.c_first_name,
        hvc.c_last_name,
        hvc.cd_gender,
        hvc.cd_marital_status,
        hvc.total_spent,
        hvc.purchase_count,
        cl.ca_city,
        cl.ca_state
    FROM high_value_customers AS hvc
    LEFT JOIN customer_location AS cl ON hvc.c_customer_sk = cl.c_customer_sk
    WHERE cl.city_rank = 1
)
SELECT 
    cwl.c_customer_sk,
    CONCAT(cwl.c_first_name, ' ', cwl.c_last_name) AS full_name,
    cwl.cd_gender,
    cwl.cd_marital_status,
    cwl.total_spent,
    CASE 
        WHEN total_spent > 10000 THEN 'High Roller'
        WHEN total_spent > 5000 THEN 'Moderate Spender'
        ELSE 'Budget Buyer'
    END AS customer_type,
    cwl.ca_city,
    cwl.ca_state,
    CASE 
        WHEN cwl.total_spent >= (SELECT AVG(total_spent) FROM high_value_customers) THEN TRUE
        ELSE FALSE
    END AS above_average_spender
FROM customer_with_locations AS cwl
WHERE cwl.total_spent IS NOT NULL
ORDER BY cwl.total_spent DESC
LIMIT 50;
