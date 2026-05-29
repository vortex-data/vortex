
WITH CustomerReturns AS (
    SELECT 
        sr_customer_sk,
        SUM(sr_return_amt) AS total_return_amt,
        COUNT(sr_ticket_number) AS return_count
    FROM 
        store_returns
    GROUP BY 
        sr_customer_sk
),
ItemSales AS (
    SELECT 
        ws_ship_customer_sk,
        SUM(ws_net_profit) AS total_net_profit,
        COUNT(ws_order_number) AS order_count
    FROM 
        web_sales
    GROUP BY 
        ws_ship_customer_sk
),
SalesSummary AS (
    SELECT 
        c.c_customer_sk,
        c.c_first_name,
        c.c_last_name,
        COALESCE(cr.total_return_amt, 0) AS total_return_amt,
        COALESCE(cr.return_count, 0) AS return_count,
        COALESCE(isales.total_net_profit, 0) AS total_net_profit,
        COALESCE(isales.order_count, 0) AS order_count
    FROM 
        customer c
    LEFT JOIN 
        CustomerReturns cr ON c.c_customer_sk = cr.sr_customer_sk
    LEFT JOIN 
        ItemSales isales ON c.c_customer_sk = isales.ws_ship_customer_sk
)
SELECT 
    s.c_customer_sk,
    s.c_first_name,
    s.c_last_name,
    s.total_return_amt,
    s.return_count,
    s.total_net_profit,
    s.order_count,
    CASE 
        WHEN s.total_net_profit > s.total_return_amt THEN 'Profitable'
        WHEN s.total_net_profit < s.total_return_amt THEN 'Unprofitable'
        ELSE 'Break Even'
    END AS profitability_status
FROM 
    SalesSummary s
WHERE 
    (s.total_return_amt > 1000 OR s.total_net_profit > 5000)
ORDER BY 
    profitability_status DESC,
    total_net_profit DESC
LIMIT 100;
