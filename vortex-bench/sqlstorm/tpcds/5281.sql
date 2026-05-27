
WITH sales_summary AS (
    SELECT 
        d.d_year, 
        d.d_month_seq, 
        d.d_quarter_seq, 
        SUM(ws.ws_net_paid) AS total_sales, 
        COUNT(DISTINCT ws.ws_order_number) AS total_orders,
        COUNT(DISTINCT ws.ws_bill_customer_sk) AS unique_customers
    FROM 
        web_sales AS ws
    JOIN 
        date_dim AS d ON ws.ws_sold_date_sk = d.d_date_sk
    JOIN 
        customer AS c ON ws.ws_bill_customer_sk = c.c_customer_sk
    JOIN 
        customer_demographics AS cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
    WHERE 
        d.d_year = 2023
        AND cd.cd_gender = 'F'
        AND cd.cd_marital_status = 'M'
    GROUP BY 
        d.d_year, d.d_month_seq, d.d_quarter_seq
), avg_sales AS (
    SELECT 
        d_year, 
        d_month_seq, 
        d_quarter_seq, 
        total_sales, 
        total_orders, 
        unique_customers,
        total_sales / NULLIF(total_orders, 0) AS avg_order_value,
        unique_customers / NULLIF(total_orders, 0) AS avg_customers_per_order
    FROM 
        sales_summary
)

SELECT 
    d_year, 
    d_month_seq, 
    d_quarter_seq, 
    total_sales, 
    total_orders, 
    unique_customers, 
    avg_order_value, 
    avg_customers_per_order
FROM 
    avg_sales
ORDER BY 
    d_year, 
    d_quarter_seq, 
    d_month_seq;
