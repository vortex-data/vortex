
WITH RECURSIVE sales_totals AS (
    SELECT
        ws_sold_date_sk,
        ws_item_sk,
        SUM(ws_ext_sales_price) AS total_sales,
        ROW_NUMBER() OVER (PARTITION BY ws_item_sk ORDER BY ws_sold_date_sk DESC) AS rnk
    FROM
        web_sales
    GROUP BY
        ws_sold_date_sk, ws_item_sk
),
customer_sales AS (
    SELECT
        c.c_customer_sk,
        c.c_first_name,
        c.c_last_name,
        COUNT(DISTINCT ws.ws_order_number) AS web_order_count,
        AVG(COALESCE(ws.ws_net_paid, 0)) AS avg_net_paid
    FROM
        customer c
    LEFT JOIN web_sales ws ON c.c_customer_sk = ws.ws_bill_customer_sk
    GROUP BY
        c.c_customer_sk, c.c_first_name, c.c_last_name
),
latest_shipping AS (
    SELECT
        ws_ship_customer_sk,
        sm.sm_type,
        COUNT(DISTINCT ws_order_number) AS order_count
    FROM
        web_sales
    JOIN ship_mode sm ON ws_ship_mode_sk = sm.sm_ship_mode_sk
    GROUP BY
        ws_ship_customer_sk, sm.sm_type
),
inventory_summary AS (
    SELECT
        inv_item_sk,
        SUM(inv_quantity_on_hand) AS total_inventory
    FROM
        inventory
    GROUP BY
        inv_item_sk
)
SELECT
    cs.c_customer_sk,
    cs.c_first_name,
    cs.c_last_name,
    lt.sm_type,
    lt.order_count,
    COALESCE(st.total_sales, 0) AS web_sales_total,
    inv.total_inventory
FROM
    customer_sales cs
LEFT JOIN latest_shipping lt ON cs.c_customer_sk = lt.ws_ship_customer_sk
LEFT JOIN sales_totals st ON cs.c_customer_sk = st.ws_item_sk
LEFT JOIN inventory_summary inv ON cs.c_customer_sk = inv.inv_item_sk
WHERE
    cs.web_order_count > 5
    AND (lt.order_count IS NULL OR lt.order_count > 2)
ORDER BY
    cs.c_last_name, cs.c_first_name;
