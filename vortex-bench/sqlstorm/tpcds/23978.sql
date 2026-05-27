
WITH RankedCustomers AS (
    SELECT
        c.c_customer_sk,
        c.c_customer_id,
        cd.cd_gender,
        cd.cd_marital_status,
        cd.cd_purchase_estimate,
        ROW_NUMBER() OVER (PARTITION BY cd.cd_gender ORDER BY cd.cd_purchase_estimate DESC) AS rn
    FROM
        customer c
    JOIN
        customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
),
InventoryAnalysis AS (
    SELECT
        inv.inv_item_sk,
        SUM(inv.inv_quantity_on_hand) AS total_quantity,
        COUNT(DISTINCT inv.inv_warehouse_sk) AS warehouse_count
    FROM
        inventory inv
    GROUP BY
        inv.inv_item_sk
),
SalesData AS (
    SELECT
        ws.ws_item_sk,
        SUM(ws.ws_net_profit) AS total_net_profit
    FROM
        web_sales ws
    WHERE
        ws.ws_sold_date_sk >= (SELECT MAX(d.d_date_sk) FROM date_dim d WHERE d.d_year = 2022)
    GROUP BY
        ws.ws_item_sk
)
SELECT
    rc.c_customer_id,
    rc.cd_gender,
    ia.total_quantity,
    sd.total_net_profit,
    COALESCE(sd.total_net_profit, 0) - COALESCE(ia.total_quantity * 0.5, 0) AS adjusted_profit
FROM
    RankedCustomers rc
LEFT JOIN
    InventoryAnalysis ia ON ia.inv_item_sk = rc.c_customer_sk
FULL OUTER JOIN
    SalesData sd ON sd.ws_item_sk = rc.c_customer_sk
WHERE
    rc.rn <= 5 AND
    (rc.cd_marital_status = 'M' OR rc.cd_purchase_estimate IS NOT NULL) AND
    (ia.total_quantity IS NULL OR ia.warehouse_count > 2)
ORDER BY
    adjusted_profit DESC, rc.c_customer_id;
