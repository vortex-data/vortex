
WITH SalesSummary AS (
    SELECT 
        c.c_customer_id AS customer_id,
        SUM(ws.ws_ext_sales_price) AS total_sales,
        AVG(ws.ws_net_profit) AS avg_profit,
        COUNT(ws.ws_order_number) AS order_count,
        d.d_year,
        d.d_month_seq
    FROM 
        web_sales ws
    JOIN 
        customer c ON ws.ws_bill_customer_sk = c.c_customer_sk
    JOIN 
        date_dim d ON ws.ws_sold_date_sk = d.d_date_sk
    WHERE 
        d.d_year = 2022
    GROUP BY 
        c.c_customer_id, d.d_year, d.d_month_seq
),
RankedSales AS (
    SELECT 
        customer_id,
        total_sales,
        avg_profit,
        order_count,
        ROW_NUMBER() OVER (PARTITION BY d_year, d_month_seq ORDER BY total_sales DESC) AS sales_rank,
        d_year,
        d_month_seq
    FROM 
        SalesSummary
)
SELECT 
    r.customer_id,
    r.total_sales,
    r.avg_profit,
    r.order_count,
    d.d_month_seq,
    d.d_year
FROM 
    RankedSales r
JOIN 
    date_dim d ON r.d_year = d.d_year AND r.d_month_seq = d.d_month_seq
WHERE 
    r.sales_rank <= 10
ORDER BY 
    d.d_year, d.d_month_seq, r.total_sales DESC;
