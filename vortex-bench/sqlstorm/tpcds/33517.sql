
WITH RECURSIVE SalesCTE AS (
    SELECT 
        ws_item_sk,
        COUNT(ws_order_number) AS total_sales,
        SUM(ws_ext_sales_price) AS total_revenue
    FROM 
        web_sales
    GROUP BY 
        ws_item_sk
    UNION ALL
    SELECT 
        cs_item_sk,
        COUNT(cs_order_number) AS total_sales,
        SUM(cs_ext_sales_price) AS total_revenue
    FROM 
        catalog_sales
    GROUP BY 
        cs_item_sk
),
SalesSummary AS (
    SELECT 
        item.i_item_id,
        item.i_item_desc,
        COALESCE(SUM(s.total_sales), 0) AS total_sales,
        COALESCE(SUM(s.total_revenue), 0) AS total_revenue
    FROM 
        item 
    LEFT JOIN 
        SalesCTE s ON item.i_item_sk = s.ws_item_sk
    GROUP BY 
        item.i_item_id,
        item.i_item_desc
),
CustomerData AS (
    SELECT 
        c.c_customer_sk,
        d.d_year,
        cd.cd_gender,
        SUM(ws.ws_net_paid) AS total_spending
    FROM 
        customer c
    JOIN 
        web_sales ws ON c.c_customer_sk = ws.ws_bill_customer_sk
    JOIN 
        customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
    JOIN 
        date_dim d ON ws.ws_sold_date_sk = d.d_date_sk
    WHERE 
        d.d_year >= 2021
    GROUP BY 
        c.c_customer_sk,
        d.d_year,
        cd.cd_gender
),
HighValueCustomers AS (
    SELECT 
        cd.d_year,
        cd.cd_gender,
        COUNT(DISTINCT cd.c_customer_sk) AS high_value_count
    FROM 
        CustomerData cd
    WHERE 
        cd.total_spending > (SELECT AVG(total_spending) FROM CustomerData)
    GROUP BY 
        cd.d_year,
        cd.cd_gender
)
SELECT 
    ss.i_item_id,
    ss.i_item_desc,
    ss.total_sales,
    ss.total_revenue,
    hvc.d_year,
    hvc.cd_gender,
    hvc.high_value_count
FROM 
    SalesSummary ss
LEFT JOIN 
    HighValueCustomers hvc ON ss.total_sales > 100
ORDER BY 
    ss.total_revenue DESC, 
    hvc.high_value_count DESC
LIMIT 50;
