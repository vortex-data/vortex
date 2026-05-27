
WITH RECURSIVE sales_hierarchy AS (
    SELECT 
        c_customer_sk,
        c_first_name,
        c_last_name,
        c_birth_year,
        0 AS level,
        CAST(c_first_name AS VARCHAR(100)) AS path
    FROM customer
    WHERE c_birth_year IS NOT NULL

    UNION ALL

    SELECT 
        s.ss_customer_sk,
        c.c_first_name,
        c.c_last_name,
        c.c_birth_year,
        sh.level + 1,
        CONCAT(sh.path, ' -> ', c.c_first_name)
    FROM store_sales s
    JOIN customer c ON s.ss_customer_sk = c.c_customer_sk
    JOIN sales_hierarchy sh ON sh.c_customer_sk = s.ss_customer_sk
    WHERE sh.level < 2
), 
inventory_summary AS (
    SELECT 
        inv.inv_item_sk,
        SUM(inv.inv_quantity_on_hand) AS total_quantity,
        RANK() OVER (ORDER BY SUM(inv.inv_quantity_on_hand) DESC) AS rank
    FROM inventory inv
    GROUP BY inv.inv_item_sk
),
web_sales_summary AS (
    SELECT 
        ws.ws_item_sk,
        SUM(ws.ws_net_profit) AS total_net_profit,
        COUNT(DISTINCT ws.ws_order_number) AS total_orders
    FROM web_sales ws
    GROUP BY ws.ws_item_sk
),
combined_sales AS (
    SELECT 
        w.ws_item_sk,
        COALESCE(i.total_quantity, 0) AS total_quantity,
        w.total_net_profit,
        w.total_orders
    FROM web_sales_summary w
    LEFT JOIN inventory_summary i ON w.ws_item_sk = i.inv_item_sk
)
SELECT 
    sh.level,
    sh.path,
    cs.c_customer_sk,
    cs.c_first_name,
    cs.c_last_name,
    cs.c_birth_year,
    cs.c_customer_id,
    cs.c_email_address,
    cs.c_salutation,
    cs.c_preferred_cust_flag,
    cs.c_birth_country,
    SUM(cs.c_birth_year) OVER (PARTITION BY cs.c_customer_sk) AS total_birth_years,
    cs.c_login,
    cs.c_current_cdemo_sk,
    cs.c_current_addr_sk,
    cs.c_first_shipto_date_sk,
    cs.c_first_sales_date_sk,
    cs.c_last_review_date_sk,
    cs.c_current_hdemo_sk,
    COALESCE(SUM(cs.c_current_cdemo_sk), 0) AS demo_count
FROM sales_hierarchy sh
JOIN customer cs ON sh.c_customer_sk = cs.c_customer_sk
LEFT JOIN combined_sales cs_totals ON cs_totals.ws_item_sk = cs.c_current_cdemo_sk
WHERE sh.level <= 1
  AND (cs.c_birth_year IS NOT NULL OR cs.c_email_address IS NOT NULL)
GROUP BY 
    sh.level,
    sh.path,
    cs.c_customer_sk,
    cs.c_first_name,
    cs.c_last_name,
    cs.c_birth_year,
    cs.c_customer_id,
    cs.c_email_address,
    cs.c_salutation,
    cs.c_preferred_cust_flag,
    cs.c_birth_country,
    cs.c_login,
    cs.c_current_cdemo_sk,
    cs.c_current_addr_sk,
    cs.c_first_shipto_date_sk,
    cs.c_first_sales_date_sk,
    cs.c_last_review_date_sk,
    cs.c_current_hdemo_sk
ORDER BY sh.level, cs.c_customer_sk;
