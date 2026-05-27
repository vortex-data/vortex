
WITH SalesSummary AS (
    SELECT 
        ws.ws_bill_customer_sk,
        SUM(ws.ws_ext_sales_price) AS total_sales,
        COUNT(DISTINCT ws.ws_order_number) AS order_count,
        AVG(ws.ws_net_profit) AS avg_net_profit,
        MAX(ws.ws_sales_price) AS max_sales_price,
        MIN(ws.ws_sales_price) AS min_sales_price,
        COUNT(DISTINCT ws.ws_ship_mode_sk) AS distinct_shipping_methods,
        cd.cd_gender,
        cd.cd_marital_status
    FROM 
        web_sales ws
    JOIN 
        customer c ON ws.ws_bill_customer_sk = c.c_customer_sk
    JOIN 
        customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
    WHERE 
        EXISTS (SELECT 1 FROM store s WHERE s.s_store_sk = ws.ws_warehouse_sk AND s.s_state = 'CA')
      AND 
        ws.ws_sold_date_sk BETWEEN (SELECT d_date_sk FROM date_dim WHERE d_date = '2023-01-01') 
        AND (SELECT d_date_sk FROM date_dim WHERE d_date = '2023-12-31')
    GROUP BY 
        ws.ws_bill_customer_sk, cd.cd_gender, cd.cd_marital_status
),
RankedSales AS (
    SELECT 
        *,
        RANK() OVER (PARTITION BY cd_gender ORDER BY total_sales DESC) AS sales_rank
    FROM 
        SalesSummary
)
SELECT 
    s.ws_bill_customer_sk,
    s.total_sales,
    s.order_count,
    s.avg_net_profit,
    s.max_sales_price,
    s.min_sales_price,
    s.distinct_shipping_methods,
    s.cd_gender,
    s.cd_marital_status
FROM 
    RankedSales s
WHERE 
    s.sales_rank <= 10
ORDER BY 
    s.cd_gender, s.total_sales DESC;
