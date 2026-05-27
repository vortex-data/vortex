
WITH RECURSIVE customer_tree AS (
    SELECT c_customer_sk, c_first_name, c_last_name, c_current_addr_sk, 1 AS depth
    FROM customer
    WHERE c_customer_sk IS NOT NULL
    UNION ALL
    SELECT c.c_customer_sk, c.c_first_name, c.c_last_name, c.c_current_addr_sk, ct.depth + 1
    FROM customer c
    JOIN customer_tree ct ON c.c_current_addr_sk = ct.c_current_addr_sk
    WHERE c.c_customer_sk <> ct.c_customer_sk AND ct.depth < 5
),
item_summary AS (
    SELECT i.i_item_sk, i.i_item_id, SUM(ws.ws_quantity) AS total_quantity_sold
    FROM item i
    LEFT JOIN web_sales ws ON i.i_item_sk = ws.ws_item_sk
    GROUP BY i.i_item_sk, i.i_item_id
),
store_sales_summary AS (
    SELECT ss.ss_store_sk, SUM(ss.ss_net_profit) AS total_net_profit, COUNT(DISTINCT ss.ss_ticket_number) AS total_sales_count
    FROM store_sales ss
    LEFT JOIN store s ON ss.ss_store_sk = s.s_store_sk
    WHERE s.s_state = 'CA' AND ss.ss_sold_date_sk >= 2459000
    GROUP BY ss.ss_store_sk
)
SELECT ct.c_first_name, ct.c_last_name, 
       COALESCE(ss.total_net_profit, 0) AS net_profit,
       COALESCE(iss.total_quantity_sold, 0) AS quantity_sold,
       CASE 
           WHEN COALESCE(ss.total_net_profit, 0) > 1000 THEN 'High Profit'
           WHEN COALESCE(ss.total_net_profit, 0) BETWEEN 500 AND 1000 THEN 'Moderate Profit'
           ELSE 'Low Profit'
       END AS profit_category
FROM customer_tree ct
LEFT JOIN store_sales_summary ss ON ct.c_current_addr_sk = ss.ss_store_sk
LEFT JOIN item_summary iss ON ct.c_customer_sk = iss.i_item_sk
WHERE EXISTS (
    SELECT 1
    FROM customer_demographics cd
    WHERE cd.cd_demo_sk = ct.c_customer_sk
    AND cd.cd_marital_status = 'M'
)
ORDER BY ct.depth DESC, ct.c_last_name
LIMIT 50;
