
SELECT 
    c.c_first_name,
    c.c_last_name,
    SUM(ws.ws_sales_price) AS total_spent,
    COUNT(ws.ws_order_number) AS total_orders
FROM 
    customer c
JOIN 
    web_sales ws ON c.c_customer_sk = ws.ws_bill_customer_sk
WHERE 
    ws.ws_sold_date_sk BETWEEN 20200101 AND 20201231
GROUP BY 
    c.c_first_name, c.c_last_name
ORDER BY 
    total_spent DESC
LIMIT 100;
