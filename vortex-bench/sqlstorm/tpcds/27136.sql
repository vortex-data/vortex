
WITH ranked_customers AS (
    SELECT 
        c.c_customer_sk,
        c.c_first_name,
        c.c_last_name,
        cd.cd_gender,
        cd.cd_marital_status,
        COUNT(sr.sr_ticket_number) AS total_returns,
        SUM(sr.sr_return_amt) AS total_return_value,
        SUM(sr.sr_return_quantity) AS total_return_quantity,
        ROW_NUMBER() OVER (PARTITION BY cd.cd_gender ORDER BY COUNT(sr.sr_ticket_number) DESC) AS return_rank
    FROM 
        customer c
    LEFT JOIN 
        store_returns sr ON c.c_customer_sk = sr.sr_customer_sk
    LEFT JOIN 
        customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
    GROUP BY 
        c.c_customer_sk, c.c_first_name, c.c_last_name, cd.cd_gender, cd.cd_marital_status
)
SELECT 
    CONCAT(c_first_name, ' ', c_last_name) AS full_name,
    cd_gender AS gender,
    cd_marital_status AS marital_status,
    total_returns,
    total_return_value,
    total_return_quantity
FROM 
    ranked_customers
WHERE 
    return_rank <= 10
ORDER BY 
    cd_gender, total_returns DESC;
