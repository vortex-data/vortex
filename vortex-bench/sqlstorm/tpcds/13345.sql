
SELECT 
    COUNT(DISTINCT c.c_customer_id) AS unique_customers, 
    SUM(ss.ss_net_paid) AS total_sales, 
    AVG(ss.ss_sales_price) AS average_sales_price 
FROM 
    customer c 
JOIN 
    store_sales ss ON c.c_customer_sk = ss.ss_customer_sk 
JOIN 
    date_dim d ON ss.ss_sold_date_sk = d.d_date_sk 
WHERE 
    d.d_year = 2023 
GROUP BY 
    d.d_month_seq 
ORDER BY 
    d.d_month_seq;
