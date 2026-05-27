
SELECT c.c_customer_id, SUM(ss.ss_quantity) AS total_quantity_sold
FROM customer c
JOIN store_sales ss ON c.c_customer_sk = ss.ss_customer_sk
GROUP BY c.c_customer_id
ORDER BY total_quantity_sold DESC
LIMIT 10;
