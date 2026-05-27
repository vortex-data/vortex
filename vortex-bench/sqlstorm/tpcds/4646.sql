
WITH SalesData AS (
    SELECT
        ws_sold_date_sk,
        ws_item_sk,
        SUM(ws_quantity) AS total_quantity,
        SUM(ws_net_paid_inc_tax) AS total_sales,
        COUNT(DISTINCT ws_order_number) AS total_orders
    FROM
        web_sales
    WHERE
        ws_sold_date_sk BETWEEN (SELECT MAX(d_date_sk) FROM date_dim WHERE d_year = 2023) - 30
        AND (SELECT MAX(d_date_sk) FROM date_dim WHERE d_year = 2023)
    GROUP BY
        ws_sold_date_sk, ws_item_sk
),
ItemDetails AS (
    SELECT
        i_item_sk,
        i_item_desc,
        i_product_name,
        i_current_price,
        i_brand,
        i_class,
        i_category
    FROM
        item
),
CustomerStats AS (
    SELECT
        cd_demo_sk,
        COUNT(c_customer_sk) AS customer_count,
        AVG(cd_purchase_estimate) AS average_purchase_estimate
    FROM
        customer_demographics cd
    JOIN
        customer c ON cd.cd_demo_sk = c.c_current_cdemo_sk
    GROUP BY
        cd_demo_sk
),
RankedSales AS (
    SELECT
        sd.ws_item_sk,
        sd.total_quantity,
        sd.total_sales,
        ROW_NUMBER() OVER (ORDER BY sd.total_sales DESC) AS sales_rank
    FROM
        SalesData sd
)
SELECT
    rs.sales_rank,
    id.i_item_desc,
    id.i_product_name,
    id.i_current_price,
    rs.total_quantity,
    rs.total_sales,
    cs.customer_count,
    cs.average_purchase_estimate
FROM
    RankedSales rs
JOIN
    ItemDetails id ON rs.ws_item_sk = id.i_item_sk
LEFT JOIN
    CustomerStats cs ON id.i_item_sk = cs.cd_demo_sk
WHERE
    rs.sales_rank <= 10
ORDER BY
    rs.sales_rank;
