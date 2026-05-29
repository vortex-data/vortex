
WITH SalesData AS (
    SELECT
        ws.ws_sold_date_sk,
        ws.ws_item_sk,
        SUM(ws.ws_quantity) AS total_quantity,
        SUM(ws.ws_net_paid) AS total_sales,
        AVG(ws.ws_sales_price) AS avg_price
    FROM
        web_sales ws
    JOIN date_dim dd ON ws.ws_sold_date_sk = dd.d_date_sk
    WHERE
        dd.d_year = 2022
    GROUP BY
        ws.ws_sold_date_sk,
        ws.ws_item_sk
),
TopItems AS (
    SELECT
        sd.ws_item_sk,
        sd.total_quantity,
        sd.total_sales,
        ROW_NUMBER() OVER (ORDER BY sd.total_sales DESC) AS sales_rank
    FROM
        SalesData sd
)
SELECT
    ti.ws_item_sk,
    ti.total_quantity,
    ti.total_sales,
    i.i_item_desc,
    i.i_current_price,
    cd.cd_gender,
    cd.cd_marital_status
FROM
    TopItems ti
JOIN item i ON ti.ws_item_sk = i.i_item_sk
JOIN store_sales ss ON ti.ws_item_sk = ss.ss_item_sk
JOIN customer c ON ss.ss_customer_sk = c.c_customer_sk
JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
WHERE
    ti.sales_rank <= 10
ORDER BY
    ti.total_sales DESC;
