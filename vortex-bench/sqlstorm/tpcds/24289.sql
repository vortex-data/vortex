
WITH RankedSales AS (
    SELECT
        ws.ws_item_sk,
        ws.ws_order_number,
        ws.ws_sales_price,
        ws.ws_quantity,
        ROW_NUMBER() OVER (PARTITION BY ws.ws_item_sk ORDER BY ws.ws_sales_price DESC) AS rn
    FROM
        web_sales ws
    WHERE
        ws.ws_sales_price IS NOT NULL
),
InventoryStats AS (
    SELECT
        inv.inv_item_sk,
        SUM(inv.inv_quantity_on_hand) AS total_quantity,
        COUNT(DISTINCT inv.inv_warehouse_sk) AS distinct_warehouses
    FROM
        inventory inv
    GROUP BY
        inv.inv_item_sk
)
SELECT
    ca.ca_address_id,
    ROUND(AVG(COALESCE(r.ws_sales_price, 0)), 2) AS avg_price,
    SUM(COALESCE(i.total_quantity, 0)) AS total_inventory,
    COUNT(DISTINCT c.c_customer_id) AS number_of_customers,
    CASE
        WHEN COUNT(DISTINCT c.c_customer_id) > 0 THEN 'Active'
        ELSE 'Inactive'
    END AS customer_status
FROM
    customer_address ca
LEFT JOIN customer c ON c.c_current_addr_sk = ca.ca_address_sk
LEFT JOIN RankedSales r ON r.rn = 1 AND r.ws_item_sk = (
    SELECT
        inv.inv_item_sk
    FROM
        InventoryStats inv
    WHERE
        inv.total_quantity > 0
    ORDER BY
        inv.total_quantity DESC
    LIMIT 1
)
LEFT JOIN InventoryStats i ON r.ws_item_sk = i.inv_item_sk
GROUP BY
    ca.ca_address_id
HAVING
    AVG(r.ws_sales_price) > (
        SELECT
            AVG(ws.ws_sales_price)
        FROM
            web_sales ws
        WHERE
            ws.ws_sales_price IS NOT NULL
    )
ORDER BY
    total_inventory DESC,
    customer_status DESC
LIMIT 100;
