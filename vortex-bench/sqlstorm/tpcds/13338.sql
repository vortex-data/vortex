
SELECT 
    c.c_first_name, 
    c.c_last_name, 
    SUM(ws.ws_sales_price) AS total_sales 
FROM 
    customer c 
JOIN 
    web_sales ws ON c.c_customer_sk = ws.ws_bill_customer_sk 
GROUP BY 
    c.c_first_name, c.c_last_name 
ORDER BY 
    total_sales DESC 
LIMIT 100;
