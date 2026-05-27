
SELECT 
    c.c_customer_id, 
    SUM(ss.ss_quantity) AS total_quantity_sold, 
    SUM(ss.ss_sales_price) AS total_sales_amount
FROM 
    customer c
JOIN 
    store_sales ss ON c.c_customer_sk = ss.ss_customer_sk
JOIN 
    item i ON ss.ss_item_sk = i.i_item_sk
WHERE 
    i.i_current_price > 50.00
GROUP BY 
    c.c_customer_id
ORDER BY 
    total_sales_amount DESC
LIMIT 100;
