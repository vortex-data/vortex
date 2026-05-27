
WITH sales_summary AS (
    SELECT 
        c.c_customer_id,
        SUM(ws.ws_net_profit) AS total_net_profit,
        COUNT(DISTINCT ws.ws_order_number) AS total_orders,
        AVG(ws.ws_quantity) AS avg_quantity_per_order,
        MAX(ws.ws_sales_price) AS max_sales_price,
        MIN(ws.ws_sales_price) AS min_sales_price
    FROM 
        customer c
    JOIN 
        web_sales ws ON c.c_customer_sk = ws.ws_bill_customer_sk
    JOIN 
        date_dim d ON ws.ws_sold_date_sk = d.d_date_sk
    WHERE 
        d.d_year = 2023
        AND c.c_current_addr_sk IS NOT NULL
    GROUP BY 
        c.c_customer_id
),
customer_demographics AS (
    SELECT 
        cd.cd_demo_sk,
        cd.cd_gender,
        cd.cd_marital_status,
        cd.cd_education_status,
        SUM(ss.total_net_profit) AS total_profit_by_demo
    FROM 
        customer_demographics cd
    JOIN 
        customer c ON cd.cd_demo_sk = c.c_current_cdemo_sk
    JOIN 
        sales_summary ss ON c.c_customer_id = ss.c_customer_id
    GROUP BY 
        cd.cd_demo_sk, cd.cd_gender, cd.cd_marital_status, cd.cd_education_status
),
top_customers AS (
    SELECT 
        cd.cd_demo_sk,
        cd.cd_gender,
        cd.cd_marital_status,
        cd.cd_education_status,
        cd.total_profit_by_demo,
        RANK() OVER (ORDER BY cd.total_profit_by_demo DESC) AS rank
    FROM 
        customer_demographics cd
)
SELECT 
    tc.rank,
    tc.cd_gender,
    tc.cd_marital_status,
    tc.cd_education_status,
    tc.total_profit_by_demo
FROM 
    top_customers tc
WHERE 
    tc.rank <= 10
ORDER BY 
    tc.rank;
