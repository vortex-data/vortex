
WITH RECURSIVE sales_hierarchy AS (
    SELECT 
        c.c_customer_id,
        c.c_current_cdemo_sk,
        c.c_first_name,
        c.c_last_name,
        cd.cd_gender,
        cd.cd_marital_status,
        SUM(ss.ss_net_profit) AS total_sales
    FROM 
        customer AS c
    LEFT JOIN 
        customer_demographics AS cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
    LEFT JOIN 
        store_sales AS ss ON c.c_customer_sk = ss.ss_customer_sk
    GROUP BY 
        c.c_customer_id, c.c_current_cdemo_sk, c.c_first_name, c.c_last_name, cd.cd_gender, cd.cd_marital_status
    HAVING 
        SUM(ss.ss_net_profit) > 1000
),
recent_sales AS (
    SELECT 
        ws.ws_order_number,
        ws.ws_sold_date_sk,
        ws.ws_item_sk,
        ws.ws_net_paid,
        w.w_warehouse_name
    FROM 
        web_sales AS ws
    JOIN 
        warehouse AS w ON ws.ws_warehouse_sk = w.w_warehouse_sk
    WHERE 
        ws.ws_sold_date_sk >= (
            SELECT MAX(d.d_date_sk) - 30 
            FROM date_dim AS d
        )
),
promotions AS (
    SELECT 
        p.p_promo_name,
        p.p_discount_active,
        COUNT(DISTINCT ws.ws_order_number) AS order_count
    FROM 
        promotion AS p
    JOIN 
        web_sales AS ws ON p.p_promo_sk = ws.ws_promo_sk
    WHERE 
        p.p_discount_active = 'Y'
    GROUP BY 
        p.p_promo_name, p.p_discount_active
    ORDER BY 
        order_count DESC
)
SELECT 
    sh.c_first_name,
    sh.c_last_name,
    sh.cd_gender,
    sh.total_sales,
    rs.ws_order_number,
    rs.ws_net_paid,
    pr.p_promo_name,
    pr.order_count
FROM 
    sales_hierarchy AS sh
LEFT JOIN 
    recent_sales AS rs ON sh.c_current_cdemo_sk = rs.ws_item_sk
LEFT JOIN 
    promotions AS pr ON rs.ws_order_number = pr.order_count
WHERE 
    sh.total_sales IS NOT NULL
ORDER BY 
    sh.total_sales DESC
LIMIT 100;
