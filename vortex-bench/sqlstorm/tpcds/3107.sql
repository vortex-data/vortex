
WITH RECURSIVE Revenue_CTE AS (
    SELECT 
        ws.ws_item_sk,
        SUM(ws.ws_ext_sales_price) AS total_revenue,
        COUNT(DISTINCT ws.ws_order_number) AS total_sales,
        DENSE_RANK() OVER (ORDER BY SUM(ws.ws_ext_sales_price) DESC) AS rank
    FROM 
        web_sales ws
    JOIN 
        date_dim d ON ws.ws_sold_date_sk = d.d_date_sk
    WHERE 
        d.d_year = 2023
    GROUP BY 
        ws.ws_item_sk
), 
Customer_Stats AS (
    SELECT 
        c.c_customer_sk,
        cd.cd_gender,
        COUNT(DISTINCT ws.ws_order_number) AS total_orders,
        COALESCE(SUM(ws.ws_net_paid), 0) AS total_spent,
        SUM(CASE WHEN ws.ws_ship_mode_sk IS NOT NULL THEN ws.ws_quantity ELSE 0 END) AS shipped_quantity
    FROM 
        customer c
    LEFT JOIN 
        web_sales ws ON c.c_customer_sk = ws.ws_bill_customer_sk
    LEFT JOIN 
        customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
    GROUP BY 
        c.c_customer_sk, cd.cd_gender
)
SELECT 
    cs.c_customer_sk,
    cs.cd_gender,
    cs.total_orders,
    cs.total_spent,
    rc.total_revenue,
    rc.total_sales
FROM 
    Customer_Stats cs
LEFT JOIN 
    Revenue_CTE rc ON cs.c_customer_sk = rc.ws_item_sk
WHERE 
    cs.total_orders > 0
    AND rc.total_revenue IS NOT NULL
ORDER BY 
    cs.total_spent DESC, rc.total_revenue DESC
LIMIT 100;
