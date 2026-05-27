
WITH RECURSIVE sales_summary AS (
    SELECT 
        ws_bill_customer_sk,
        SUM(ws_net_profit) AS total_net_profit,
        COUNT(ws_order_number) AS order_count,
        ROW_NUMBER() OVER (PARTITION BY ws_bill_customer_sk ORDER BY SUM(ws_net_profit) DESC) AS rn
    FROM 
        web_sales
    GROUP BY 
        ws_bill_customer_sk
),
customer_details AS (
    SELECT 
        c.c_customer_sk,
        c.c_first_name,
        c.c_last_name,
        cd.cd_gender,
        cd.cd_marital_status,
        cd.cd_purchase_estimate,
        a.ca_state
    FROM 
        customer c
    JOIN 
        customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
    LEFT JOIN 
        customer_address a ON c.c_current_addr_sk = a.ca_address_sk
)
SELECT 
    cd.c_first_name,
    cd.c_last_name,
    cd.cd_gender,
    cd.cd_marital_status,
    COALESCE(ss.total_net_profit, 0) AS total_net_profit,
    COALESCE(ss.order_count, 0) AS order_count,
    cd.ca_state
FROM 
    customer_details cd
LEFT JOIN 
    sales_summary ss ON cd.c_customer_sk = ss.ws_bill_customer_sk AND ss.rn = 1
WHERE 
    (cd.cd_purchase_estimate > 1000 OR cd.cd_marital_status = 'M')
    AND cd.ca_state IS NOT NULL
ORDER BY 
    total_net_profit DESC
LIMIT 100;
