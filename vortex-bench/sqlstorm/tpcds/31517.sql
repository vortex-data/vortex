
WITH RECURSIVE SalesCTE AS (
    SELECT ss_sold_date_sk, ss_item_sk, ss_quantity, ss_net_paid, ss_store_sk,
           ROW_NUMBER() OVER (PARTITION BY ss_item_sk ORDER BY ss_sold_date_sk DESC) AS rn
    FROM store_sales
    WHERE ss_sold_date_sk IN (SELECT d_date_sk FROM date_dim WHERE d_year = 2022)
),
TopSales AS (
    SELECT ss_item_sk, SUM(ss_quantity) AS total_quantity, SUM(ss_net_paid) AS total_revenue
    FROM SalesCTE
    WHERE rn = 1
    GROUP BY ss_item_sk
),
CustomerReturns AS (
    SELECT sr_item_sk, SUM(sr_return_quantity) AS return_quantity, SUM(sr_return_amt) AS total_return_amt
    FROM store_returns
    WHERE sr_returned_date_sk IN (SELECT d_date_sk FROM date_dim WHERE d_year = 2022)
    GROUP BY sr_item_sk
),
CombinedSales AS (
    SELECT ts.ss_item_sk,
           ts.total_quantity,
           ts.total_revenue,
           COALESCE(cr.return_quantity, 0) AS total_return_quantity,
           COALESCE(cr.total_return_amt, 0) AS total_return_amount,
           (ts.total_revenue - COALESCE(cr.total_return_amt, 0)) AS net_revenue
    FROM TopSales ts
    LEFT JOIN CustomerReturns cr ON ts.ss_item_sk = cr.sr_item_sk
),
RankedSales AS (
    SELECT *, 
           RANK() OVER (ORDER BY net_revenue DESC) AS revenue_rank
    FROM CombinedSales
)
SELECT i.i_item_id, i.i_item_desc, cs.total_quantity, cs.total_revenue, 
       cs.total_return_quantity, cs.total_return_amount, cs.net_revenue, cs.revenue_rank
FROM RankedSales cs
JOIN item i ON cs.ss_item_sk = i.i_item_sk
WHERE cs.revenue_rank <= 10;
