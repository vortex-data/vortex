
WITH SalesData AS (
    SELECT
        ws_item_sk,
        SUM(ws_quantity) AS total_quantity,
        SUM(ws_ext_sales_price) AS total_sales,
        AVG(ws_net_paid) AS average_payment,
        COUNT(DISTINCT ws_order_number) AS order_count
    FROM web_sales
    WHERE ws_sold_date_sk BETWEEN 2450000 AND 2451000
    GROUP BY ws_item_sk
),
FilteredSales AS (
    SELECT
        sd.ws_item_sk,
        sd.total_quantity,
        sd.total_sales,
        sd.average_payment
    FROM SalesData sd
    JOIN item i ON sd.ws_item_sk = i.i_item_sk
    WHERE i.i_current_price IS NOT NULL
        AND i.i_formulation = 'Liquid'
        AND sd.total_quantity > (
            SELECT AVG(total_quantity)
            FROM SalesData
        )
),
TopItems AS (
    SELECT
        fs.ws_item_sk,
        ROW_NUMBER() OVER (ORDER BY fs.total_sales DESC) AS rank
    FROM FilteredSales fs
)
SELECT
    i.i_item_id,
    COALESCE(SUM(ws.ws_net_paid_inc_tax), 0) AS total_net_paid_inc_tax,
    COALESCE(COUNT(ws.ws_order_number), 0) AS order_total,
    MAX(i.i_current_price) AS max_price,
    COUNT(DISTINCT CASE WHEN ws.ws_ext_tax > 0 THEN ws.ws_order_number END) AS orders_with_tax,
    STRING_AGG(DISTINCT i.i_brand, ', ') AS brands_utilized
FROM item i
LEFT JOIN web_sales ws ON i.i_item_sk = ws.ws_item_sk
JOIN TopItems ti ON i.i_item_sk = ti.ws_item_sk
WHERE ti.rank <= 10
GROUP BY i.i_item_id
ORDER BY total_net_paid_inc_tax DESC;
