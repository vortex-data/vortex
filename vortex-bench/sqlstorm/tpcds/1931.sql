WITH CustomerReturns AS (
    SELECT 
        cr_returning_customer_sk,
        SUM(cr_return_amount) AS total_return_amount,
        COUNT(DISTINCT cr_order_number) AS total_orders_returned,
        AVG(cr_return_quantity) AS avg_return_quantity
    FROM 
        catalog_returns
    GROUP BY 
        cr_returning_customer_sk
),
WebSalesAnalysis AS (
    SELECT 
        ws_ship_customer_sk,
        SUM(ws_sales_price) AS total_sales,
        COUNT(ws_order_number) AS total_orders,
        ROW_NUMBER() OVER (PARTITION BY ws_ship_customer_sk ORDER BY SUM(ws_sales_price) DESC) AS sales_rank
    FROM 
        web_sales
    WHERE 
        ws_sold_date_sk BETWEEN 2450000 AND 2450600 
    GROUP BY 
        ws_ship_customer_sk
),
Analysis AS (
    SELECT 
        c.c_customer_sk,
        c.c_first_name,
        c.c_last_name,
        COALESCE(cr.total_return_amount, 0) AS total_return_amount,
        COALESCE(ws.total_sales, 0) AS total_sales,
        CASE 
            WHEN COALESCE(ws.total_sales, 0) = 0 THEN NULL 
            ELSE ROUND(COALESCE(cr.total_return_amount, 0) / COALESCE(ws.total_sales, 0), 2) 
        END AS return_to_sales_ratio
    FROM 
        customer c
    LEFT JOIN 
        CustomerReturns cr ON c.c_customer_sk = cr.cr_returning_customer_sk
    LEFT JOIN 
        WebSalesAnalysis ws ON c.c_customer_sk = ws.ws_ship_customer_sk
)
SELECT 
    a.c_customer_sk,
    a.c_first_name,
    a.c_last_name,
    a.total_return_amount,
    a.total_sales,
    a.return_to_sales_ratio,
    CASE 
        WHEN a.return_to_sales_ratio IS NULL OR a.return_to_sales_ratio > 0.5 THEN 'High Return'
        WHEN a.return_to_sales_ratio <= 0.5 AND a.return_to_sales_ratio > 0 THEN 'Moderate Return'
        ELSE 'No Returns'
    END AS return_category
FROM 
    Analysis a
WHERE 
    a.total_sales > 1000 
ORDER BY 
    a.total_sales DESC
LIMIT 100;