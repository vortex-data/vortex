
WITH RECURSIVE CustomerReturns AS (
    SELECT
        cr_returning_customer_sk,
        SUM(cr_return_quantity) AS total_return_quantity,
        SUM(cr_return_amount) AS total_return_amount
    FROM
        catalog_returns
    WHERE
        cr_returned_date_sk IN (SELECT d_date_sk FROM date_dim WHERE d_year = 2023)
    GROUP BY
        cr_returning_customer_sk
), InventoryLevels AS (
    SELECT
        inv_warehouse_sk,
        SUM(inv_quantity_on_hand) AS total_quantity_on_hand
    FROM
        inventory
    GROUP BY
        inv_warehouse_sk
), WeeklySales AS (
    SELECT
        ws_bill_customer_sk,
        SUM(ws_net_paid) AS total_spent,
        EXTRACT(week FROM d_date) AS week_number
    FROM
        web_sales
    JOIN
        date_dim ON ws_sold_date_sk = d_date_sk
    WHERE
        d_year = 2023
    GROUP BY
        ws_bill_customer_sk, week_number
), PromotionStats AS (
    SELECT
        p.p_promo_sk,
        COUNT(ws_order_number) AS total_orders,
        SUM(ws_ext_discount_amt) AS total_discount
    FROM
        web_sales ws
    JOIN
        promotion p ON ws.ws_promo_sk = p.p_promo_sk
    GROUP BY
        p.p_promo_sk
)
SELECT
    c.c_first_name,
    c.c_last_name,
    cd.cd_gender,
    ABS(COALESCE(cr.total_return_quantity, 0)) AS total_return_quantity,
    COALESCE(ws.total_spent, 0) AS total_spent,
    COALESCE(il.total_quantity_on_hand, 0) AS current_inventory,
    p.total_orders,
    p.total_discount
FROM
    customer c
LEFT JOIN
    customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
LEFT JOIN
    CustomerReturns cr ON c.c_customer_sk = cr.cr_returning_customer_sk
LEFT JOIN
    WeeklySales ws ON c.c_customer_sk = ws.ws_bill_customer_sk
LEFT JOIN
    InventoryLevels il ON il.inv_warehouse_sk = (SELECT w.w_warehouse_sk FROM warehouse w LIMIT 1) 
LEFT JOIN
    PromotionStats p ON p.p_promo_sk = (SELECT MIN(promo.p_promo_sk) FROM promotion promo)
WHERE
    (cd.cd_gender = 'F' AND c.c_birth_month = 5) OR
    (cd.cd_gender = 'M' AND c.c_birth_month != 5)
GROUP BY
    c.c_first_name,
    c.c_last_name,
    cd.cd_gender,
    cr.total_return_quantity,
    ws.total_spent,
    il.total_quantity_on_hand,
    p.total_orders,
    p.total_discount
ORDER BY
    total_spent DESC,
    total_return_quantity ASC
LIMIT 10;
