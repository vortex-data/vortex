
WITH RECURSIVE sales_hierarchy AS (
    SELECT 
        ws_bill_customer_sk AS customer_sk,
        SUM(ws_ext_sales_price) AS total_sales,
        COUNT(ws_order_number) AS order_count,
        0 AS level
    FROM web_sales
    GROUP BY ws_bill_customer_sk
    
    UNION ALL
    
    SELECT 
        sr_customer_sk AS customer_sk,
        SUM(sr_return_amt) AS total_sales,
        COUNT(sr_ticket_number) AS order_count,
        1 AS level
    FROM store_returns
    GROUP BY sr_customer_sk
),
total_sales AS (
    SELECT 
        c.c_customer_sk,
        COALESCE(SUM(sh.total_sales), 0) AS total_sales,
        COUNT(DISTINCT sh.order_count) AS total_orders
    FROM customer c
    LEFT JOIN sales_hierarchy sh ON c.c_customer_sk = sh.customer_sk
    GROUP BY c.c_customer_sk
),
ranked_customers AS (
    SELECT 
        c.c_customer_id,
        ts.total_sales,
        ts.total_orders,
        DENSE_RANK() OVER (ORDER BY ts.total_sales DESC) AS sales_rank
    FROM total_sales ts
    JOIN customer c ON ts.c_customer_sk = c.c_customer_sk
)
SELECT 
    rc.c_customer_id,
    rc.total_sales,
    rc.total_orders,
    CASE 
        WHEN rc.sales_rank <= 10 THEN 'Top 10%'
        WHEN rc.sales_rank <= 30 THEN 'Top 30%'
        ELSE 'Others'
    END AS customer_segment
FROM ranked_customers rc
WHERE rc.total_orders > 5
AND rc.total_sales > (
    SELECT AVG(total_sales) 
    FROM total_sales
) OR rc.total_sales IS NULL
ORDER BY rc.total_sales DESC
LIMIT 100;
