
WITH CustomerPurchaseData AS (
    SELECT 
        c.c_customer_sk,
        c.c_first_name,
        c.c_last_name,
        SUM(ws.ws_ext_sales_price) AS total_sales,
        COUNT(DISTINCT ws.ws_order_number) AS order_count,
        COUNT(DISTINCT ws.ws_web_page_sk) AS unique_pages_visited,
        ROW_NUMBER() OVER (PARTITION BY c.c_customer_sk ORDER BY SUM(ws.ws_ext_sales_price) DESC) AS rank_sales
    FROM 
        customer c
    LEFT JOIN 
        web_sales ws ON c.c_customer_sk = ws.ws_bill_customer_sk
    JOIN 
        date_dim dd ON ws.ws_sold_date_sk = dd.d_date_sk
    WHERE 
        dd.d_year = 2022
    GROUP BY 
        c.c_customer_sk, c.c_first_name, c.c_last_name
),
TopCustomers AS (
    SELECT 
        cpd.c_customer_sk,
        cpd.c_first_name,
        cpd.c_last_name,
        cpd.total_sales,
        cpd.order_count,
        cpd.unique_pages_visited
    FROM 
        CustomerPurchaseData cpd
    WHERE 
        cpd.rank_sales <= 10
),
ReturnData AS (
    SELECT 
        sr.sr_customer_sk,
        SUM(sr.sr_return_amt_inc_tax) AS total_return_amt,
        COUNT(sr.sr_ticket_number) AS returns_count
    FROM 
        store_returns sr
    GROUP BY 
        sr.sr_customer_sk
)
SELECT 
    tc.c_first_name,
    tc.c_last_name,
    tc.total_sales AS customer_total_sales,
    COALESCE(rd.total_return_amt, 0) AS total_returns,
    tc.order_count,
    rd.returns_count,
    (tc.total_sales - COALESCE(rd.total_return_amt, 0)) AS net_revenue,
    ROUND(COALESCE(rd.total_return_amt, 0) * 100 / NULLIF(tc.total_sales, 0), 2) AS return_percentage
FROM 
    TopCustomers tc
LEFT JOIN 
    ReturnData rd ON tc.c_customer_sk = rd.sr_customer_sk
ORDER BY 
    net_revenue DESC;
