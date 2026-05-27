
WITH RankedSales AS (
    SELECT 
        ws.ws_order_number,
        ws.ws_item_sk,
        ws.ws_quantity,
        ws.ws_sales_price,
        ws.ws_net_profit,
        DENSE_RANK() OVER (PARTITION BY ws.ws_item_sk ORDER BY ws.ws_sales_price DESC) AS price_rank
    FROM 
        web_sales ws
    WHERE 
        ws.ws_sold_date_sk IN (SELECT d_date_sk FROM date_dim WHERE d_year = 2023)
),
TotalReturns AS (
    SELECT 
        wr_item_sk,
        SUM(wr_return_quantity) AS total_returned_quantity,
        SUM(wr_return_amt) AS total_return_amount
    FROM 
        web_returns
    GROUP BY 
        wr_item_sk
),
SalesAndReturns AS (
    SELECT 
        r.ws_item_sk,
        r.ws_quantity,
        r.ws_sales_price,
        COALESCE(tr.total_returned_quantity, 0) AS total_returned_quantity,
        COALESCE(tr.total_return_amount, 0) AS total_return_amount
    FROM 
        RankedSales r
    LEFT JOIN 
        TotalReturns tr ON r.ws_item_sk = tr.wr_item_sk
    WHERE 
        r.price_rank = 1
)
SELECT 
    s.ws_item_sk,
    SUM(s.ws_quantity) AS total_sales_quantity,
    SUM(s.ws_sales_price) AS total_sales_revenue,
    SUM(s.total_returned_quantity) AS total_returns,
    SUM(s.total_return_amount) AS total_returned_amount,
    (SUM(ws.ws_net_profit) - SUM(s.total_return_amount)) AS net_profit
FROM 
    SalesAndReturns s
JOIN 
    web_sales ws ON s.ws_item_sk = ws.ws_item_sk
GROUP BY 
    s.ws_item_sk
HAVING 
    (SUM(ws.ws_net_profit) - SUM(s.total_return_amount)) > 0
ORDER BY 
    net_profit DESC
LIMIT 10;
