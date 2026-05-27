
WITH CustomerSales AS (
    SELECT 
        c.c_customer_sk,
        c.c_first_name,
        c.c_last_name,
        SUM(ws.ws_net_profit) AS total_sales,
        COUNT(DISTINCT ws.ws_order_number) AS order_count
    FROM customer c
    LEFT JOIN web_sales ws ON c.c_customer_sk = ws.ws_ship_customer_sk
    WHERE ws.ws_sold_date_sk BETWEEN 2450000 AND 2450600
    GROUP BY c.c_customer_sk, c.c_first_name, c.c_last_name
),

TopCustomers AS (
    SELECT 
        c.c_customer_sk,
        c.c_first_name,
        c.c_last_name,
        cs.total_sales,
        cs.order_count,
        RANK() OVER (ORDER BY cs.total_sales DESC) AS sales_rank
    FROM CustomerSales cs
    JOIN customer c ON cs.c_customer_sk = c.c_customer_sk
)

SELECT 
    tc.c_customer_sk,
    tc.c_first_name,
    tc.c_last_name,
    COALESCE(tc.total_sales, 0) AS total_sales,
    COALESCE(tc.order_count, 0) AS order_count,
    CASE 
        WHEN tc.sales_rank <= 10 THEN 'Top 10'
        ELSE 'Others'
    END AS customer_category
FROM TopCustomers tc
WHERE tc.order_count > 0

UNION ALL

SELECT 
    ca.ca_address_sk,
    'N/A' AS c_first_name,
    'N/A' AS c_last_name,
    SUM(sr.sr_return_amt) AS total_sales,
    COUNT(sr.sr_ticket_number) AS order_count,
    'Returns' AS customer_category
FROM store_returns sr
LEFT JOIN customer_address ca ON sr.sr_addr_sk = ca.ca_address_sk
WHERE sr.sr_returned_date_sk BETWEEN 2450000 AND 2450600
GROUP BY ca.ca_address_sk
ORDER BY total_sales DESC;
