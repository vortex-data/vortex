
WITH sales_summary AS (
    SELECT 
        w.w_warehouse_id,
        SUM(ws.ws_sales_price) AS total_sales,
        COUNT(DISTINCT ws.ws_order_number) AS total_orders,
        COUNT(DISTINCT ws.ws_bill_customer_sk) AS total_customers
    FROM 
        web_sales ws
    JOIN 
        warehouse w ON ws.ws_warehouse_sk = w.w_warehouse_sk
    WHERE 
        ws.ws_sold_date_sk BETWEEN 1 AND 1000
    GROUP BY 
        w.w_warehouse_id
)
SELECT 
    ss.w_warehouse_id,
    ss.total_sales,
    ss.total_orders,
    ss.total_customers,
    (ss.total_sales / NULLIF(ss.total_orders, 0)) AS avg_order_value
FROM 
    sales_summary ss
ORDER BY 
    ss.total_sales DESC;
