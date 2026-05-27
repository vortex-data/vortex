
WITH RECURSIVE sales_cte AS (
    SELECT 
        ws_sold_date_sk,
        ws_item_sk,
        SUM(ws_quantity) AS total_quantity,
        SUM(ws_net_paid) AS total_net_paid,
        ROW_NUMBER() OVER (PARTITION BY ws_item_sk ORDER BY ws_sold_date_sk DESC) AS rn
    FROM web_sales
    GROUP BY ws_sold_date_sk, ws_item_sk
), 
total_sales AS (
    SELECT 
        ws_item_sk,
        SUM(total_quantity) AS quantity_sold,
        SUM(total_net_paid) AS net_sales
    FROM sales_cte
    WHERE rn <= 10
    GROUP BY ws_item_sk
),
high_demand_items AS (
    SELECT
        i.i_item_id,
        i.i_product_name,
        t.quantity_sold,
        t.net_sales
    FROM total_sales t
    JOIN item i ON t.ws_item_sk = i.i_item_sk
    WHERE t.quantity_sold > (
        SELECT AVG(quantity_sold) FROM total_sales
    )
),
customer_data AS (
    SELECT 
        c.c_customer_id,
        cd.cd_gender,
        hd.hd_income_band_sk,
        COUNT(DISTINCT s.ss_ticket_number) AS purchases_count
    FROM customer c
    LEFT JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
    LEFT JOIN household_demographics hd ON cd.cd_demo_sk = hd.hd_demo_sk
    LEFT JOIN store_sales s ON c.c_customer_sk = s.ss_customer_sk
    GROUP BY c.c_customer_id, cd.cd_gender, hd.hd_income_band_sk
),
top_customers AS (
    SELECT 
        c.c_customer_id AS customer_id,
        SUM(cd.purchases_count) AS total_purchases
    FROM customer_data cd
    JOIN customer c ON cd.c_customer_id = c.c_customer_id
    GROUP BY c.c_customer_id
    ORDER BY total_purchases DESC
    LIMIT 5
)
SELECT hv.i_product_name,
       hv.quantity_sold,
       hv.net_sales,
       tc.customer_id,
       tc.total_purchases
FROM high_demand_items hv
JOIN top_customers tc ON hv.quantity_sold > 100
ORDER BY hv.net_sales DESC;
