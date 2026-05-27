
WITH RankedSales AS (
    SELECT 
        ws.ws_item_sk,
        ws.ws_order_number,
        ws.ws_net_profit,
        ROW_NUMBER() OVER (PARTITION BY ws.ws_item_sk ORDER BY ws.ws_net_profit DESC) AS rank_profit,
        RANK() OVER (ORDER BY ws.ws_net_profit ASC) AS rank_all_profit
    FROM 
        web_sales ws
    WHERE 
        ws.ws_net_paid > 0
        AND (ws.ws_ext_discount_amt IS NULL OR ws.ws_ext_discount_amt >= 0)
),
HighProfitItems AS (
    SELECT
        rs.ws_item_sk,
        rs.ws_order_number,
        rs.ws_net_profit
    FROM 
        RankedSales rs
    WHERE 
        rs.rank_profit <= 10
),
SalesSummary AS (
    SELECT
        ws.ws_item_sk,
        SUM(ws.ws_quantity) AS total_quantity,
        SUM(ws.ws_net_paid) AS total_net_paid,
        AVG(ws.ws_net_profit) AS avg_profit
    FROM
        web_sales ws
    JOIN
        HighProfitItems hpi ON ws.ws_item_sk = hpi.ws_item_sk
    GROUP BY
        ws.ws_item_sk
)
SELECT 
    ci.i_item_desc,
    COALESCE(ss.total_quantity, 0) AS total_quantity,
    COALESCE(ss.total_net_paid, 0.00) AS total_net_paid,
    CASE 
        WHEN ss.avg_profit IS NOT NULL THEN ROUND(ss.avg_profit, 2)
        ELSE (SELECT AVG(ws.ws_net_profit)
              FROM web_sales ws
              WHERE ws.ws_item_sk = ci.i_item_sk 
              AND ws.ws_net_profit IS NOT NULL)
    END AS avg_net_profit,
    CASE 
        WHEN (SELECT COUNT(*) FROM SalesSummary WHERE total_net_paid > 100) >= 1 THEN 'High Sellers'
        ELSE 'Low Sellers'
    END AS seller_category
FROM 
    item ci
LEFT JOIN 
    SalesSummary ss ON ci.i_item_sk = ss.ws_item_sk
ORDER BY 
    avg_net_profit DESC
LIMIT 20;
