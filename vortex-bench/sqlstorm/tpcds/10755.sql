
SELECT 
    c.c_customer_id, 
    SUM(ws.ws_quantity) AS total_quantity_sold, 
    SUM(ws.ws_sales_price * ws.ws_quantity) AS total_sales 
FROM 
    customer c 
JOIN 
    web_sales ws ON c.c_customer_sk = ws.ws_bill_customer_sk 
JOIN 
    date_dim d ON ws.ws_sold_date_sk = d.d_date_sk 
WHERE 
    d.d_year = 2023 
GROUP BY 
    c.c_customer_id 
ORDER BY 
    total_sales DESC 
LIMIT 100;
