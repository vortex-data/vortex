
WITH RECURSIVE SalesCTE AS (
    SELECT 
        ws_item_sk,
        SUM(ws_quantity) AS total_quantity,
        SUM(ws_net_paid) AS total_sales
    FROM web_sales
    WHERE ws_sold_date_sk BETWEEN 2458849 AND 2458879 
    GROUP BY ws_item_sk

    UNION ALL

    SELECT 
        s.ss_item_sk,
        SUM(s.ss_quantity) AS total_quantity,
        SUM(s.ss_net_paid) AS total_sales
    FROM store_sales s
    JOIN SalesCTE cte ON s.ss_item_sk = cte.ws_item_sk
    WHERE s.ss_sold_date_sk BETWEEN 2458849 AND 2458879 
    GROUP BY s.ss_item_sk
),

TotalCustomerReturns AS (
    SELECT
        sr_item_sk,
        SUM(sr_return_quantity) AS total_return_quantity,
        SUM(sr_return_amt_inc_tax) AS total_return_amount
    FROM store_returns
    GROUP BY sr_item_sk
),

RankedSales AS (
    SELECT
        cte.ws_item_sk,
        cte.total_quantity,
        cte.total_sales,
        COALESCE(tr.total_return_quantity, 0) AS total_return_quantity,
        COALESCE(tr.total_return_amount, 0) AS total_return_amount,
        ROW_NUMBER() OVER (ORDER BY cte.total_sales DESC) AS sales_rank
    FROM SalesCTE cte
    LEFT JOIN TotalCustomerReturns tr ON cte.ws_item_sk = tr.sr_item_sk
)

SELECT 
    i.i_item_id,
    i.i_item_desc,
    rs.total_quantity,
    rs.total_sales,
    rs.total_return_quantity,
    rs.total_return_amount,
    CASE 
        WHEN rs.total_sales > 1000 THEN 'High'
        WHEN rs.total_sales > 500 THEN 'Medium'
        ELSE 'Low'
    END AS sales_category
FROM RankedSales rs
JOIN item i ON rs.ws_item_sk = i.i_item_sk
WHERE rs.sales_rank <= 10
ORDER BY rs.total_sales DESC;
