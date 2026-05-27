
WITH RECURSIVE SalesData AS (
    SELECT ws_sold_date_sk, ws_item_sk, SUM(ws_quantity) AS total_sales, SUM(ws_net_profit) AS total_profit
    FROM web_sales
    WHERE ws_sold_date_sk IN (
        SELECT d_date_sk
        FROM date_dim
        WHERE d_year = 2023
    )
    GROUP BY ws_sold_date_sk, ws_item_sk
    UNION ALL
    SELECT cs_sold_date_sk, cs_item_sk, SUM(cs_quantity) AS total_sales, SUM(cs_net_profit) AS total_profit
    FROM catalog_sales
    WHERE cs_sold_date_sk IN (
        SELECT d_date_sk
        FROM date_dim
        WHERE d_year = 2023
    )
    GROUP BY cs_sold_date_sk, cs_item_sk
),
AggregatedSales AS (
    SELECT sd.ws_item_sk, SUM(sd.total_sales) AS yearly_sales, SUM(sd.total_profit) AS yearly_profit
    FROM SalesData sd
    GROUP BY sd.ws_item_sk
),
TopSellingItems AS (
    SELECT i.i_item_sk, i.i_item_desc, ag.yearly_sales, ag.yearly_profit,
           RANK() OVER (ORDER BY ag.yearly_sales DESC) AS sales_rank
    FROM AggregatedSales ag
    JOIN item i ON ag.ws_item_sk = i.i_item_sk
    WHERE ag.yearly_sales > 1000
)
SELECT tsi.i_item_desc, tsi.yearly_sales, tsi.yearly_profit
FROM TopSellingItems tsi
WHERE tsi.sales_rank <= 10
ORDER BY tsi.yearly_sales DESC, tsi.yearly_profit DESC;
