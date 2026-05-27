WITH sales_summary AS (
    SELECT 
        d.d_year,
        SUM(ws.ws_net_paid) AS total_sales,
        COUNT(DISTINCT ws.ws_order_number) AS total_orders,
        SUM(ws.ws_quantity) AS total_quantity,
        COUNT(DISTINCT ws.ws_ship_customer_sk) AS unique_customers
    FROM 
        web_sales ws
    JOIN 
        date_dim d ON ws.ws_sold_date_sk = d.d_date_sk
    GROUP BY 
        d.d_year
),
customer_summary AS (
    SELECT 
        cd.cd_gender,
        COUNT(DISTINCT c.c_customer_sk) AS total_customers,
        AVG(cd.cd_purchase_estimate) AS avg_purchase_estimate,
        SUM(cd.cd_dep_count) AS total_dependents
    FROM 
        customer c
    JOIN 
        customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
    GROUP BY 
        cd.cd_gender
),
top_products AS (
    SELECT 
        i.i_item_id,
        i.i_item_desc,
        SUM(ws.ws_quantity) AS total_quantity_sold
    FROM 
        item i
    JOIN 
        web_sales ws ON i.i_item_sk = ws.ws_item_sk
    GROUP BY 
        i.i_item_id, i.i_item_desc
    ORDER BY 
        total_quantity_sold DESC
    LIMIT 10
)
SELECT 
    ss.d_year,
    ss.total_sales,
    ss.total_orders,
    ss.total_quantity,
    ss.unique_customers,
    cs.cd_gender,
    cs.total_customers,
    cs.avg_purchase_estimate,
    cs.total_dependents,
    tp.i_item_id,
    tp.i_item_desc,
    tp.total_quantity_sold
FROM 
    sales_summary ss
JOIN 
    customer_summary cs ON TRUE 
JOIN 
    top_products tp ON TRUE 
ORDER BY 
    ss.d_year, cs.cd_gender, tp.total_quantity_sold DESC;