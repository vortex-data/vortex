
WITH RECURSIVE SalesCTE AS (
    SELECT 
        ws_order_number,
        ws_item_sk,
        ws_quantity,
        ws_sales_price,
        ws_net_profit,
        1 AS Level
    FROM 
        web_sales
    WHERE 
        ws_sold_date_sk BETWEEN 1 AND 1000

    UNION ALL

    SELECT 
        cs_order_number,
        cs_item_sk,
        cs_quantity,
        cs_sales_price,
        cs_net_profit,
        Level + 1
    FROM 
        catalog_sales cs
    JOIN 
        SalesCTE s ON cs.cs_order_number = s.ws_order_number
    WHERE 
        cs_order_number IS NOT NULL
),
TopCustomers AS (
    SELECT 
        c.c_customer_id,
        SUM(s.ws_net_profit) AS total_profit,
        COUNT(s.ws_order_number) AS order_count
    FROM 
        customer c
    LEFT JOIN 
        web_sales s ON c.c_customer_sk = s.ws_bill_customer_sk
    GROUP BY 
        c.c_customer_id
    HAVING 
        SUM(s.ws_net_profit) > 1000
),
SalesSummary AS (
    SELECT 
        d.d_year,
        SUM(s.ws_quantity) AS total_quantity,
        AVG(s.ws_sales_price) AS avg_sales_price,
        SUM(s.ws_net_profit) AS total_net_profit
    FROM 
        date_dim d
    JOIN 
        web_sales s ON d.d_date_sk = s.ws_sold_date_sk
    GROUP BY 
        d.d_year
)
SELECT 
    t.c_customer_id,
    t.total_profit,
    COALESCE(s.total_quantity, 0) AS total_quantity,
    COALESCE(s.avg_sales_price, 0) AS avg_sales_price,
    s.total_net_profit,
    ROW_NUMBER() OVER (ORDER BY t.total_profit DESC) AS rank
FROM 
    TopCustomers t
LEFT JOIN 
    SalesSummary s ON t.order_count = s.total_quantity
WHERE 
    t.order_count > 5
ORDER BY 
    t.total_profit DESC
LIMIT 100;
