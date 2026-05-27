
WITH recent_orders AS (
    SELECT 
        ws_order_number,
        ws_ship_date_sk,
        ws_quantity,
        ws_net_paid_inc_tax,
        ws_ext_sales_price,
        ROW_NUMBER() OVER(PARTITION BY ws_order_number ORDER BY ws_ship_date_sk DESC) AS rn
    FROM web_sales
    WHERE ws_ship_date_sk > (
        SELECT MAX(d_date_sk) - 30 
        FROM date_dim
    )
),
refunds AS (
    SELECT 
        cr_order_number,
        SUM(cr_return_quantity) AS total_returned_quantity,
        SUM(cr_return_amt_inc_tax) AS total_return_amount
    FROM catalog_returns
    GROUP BY cr_order_number
),
joined_data AS (
    SELECT 
        ro.ws_order_number,
        ro.ws_net_paid_inc_tax,
        ro.ws_quantity,
        COALESCE(r.total_returned_quantity, 0) AS total_returned_quantity,
        COALESCE(r.total_return_amount, 0) AS total_return_amount
    FROM recent_orders ro
    LEFT JOIN refunds r ON ro.ws_order_number = r.cr_order_number
    WHERE ro.rn = 1
),
final_summary AS (
    SELECT 
        SUM(ws_net_paid_inc_tax) AS total_sales,
        SUM(total_returned_quantity) AS total_quantity_returned,
        AVG(ws_net_paid_inc_tax) AS avg_order_value,
        COUNT(ws_order_number) AS total_orders,
        SUM(CASE WHEN ws_net_paid_inc_tax IS NULL THEN 1 ELSE 0 END) AS null_sales_count
    FROM joined_data
)
SELECT 
    *,
    total_sales - total_quantity_returned AS net_sales,
    total_orders * 1.0 / NULLIF(total_sales, 0) AS order_sales_ratio
FROM final_summary
WHERE avg_order_value IS NOT NULL
ORDER BY net_sales DESC;
