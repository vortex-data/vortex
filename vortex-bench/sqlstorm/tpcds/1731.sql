
WITH ranked_sales AS (
    SELECT 
        cs_item_sk,
        SUM(cs_quantity) AS total_quantity,
        SUM(cs_net_profit) AS total_net_profit,
        DENSE_RANK() OVER (PARTITION BY cs_item_sk ORDER BY SUM(cs_net_profit) DESC) AS profit_rank
    FROM 
        catalog_sales
    GROUP BY 
        cs_item_sk
),
customer_summary AS (
    SELECT 
        c.c_customer_sk,
        COUNT(DISTINCT cs.cs_order_number) AS total_orders,
        SUM(cs.cs_net_profit) AS total_spent,
        AVG(cs.cs_sales_price) AS avg_order_value
    FROM 
        customer c
    LEFT JOIN 
        store_sales ss ON c.c_customer_sk = ss.ss_customer_sk
    LEFT JOIN 
        catalog_sales cs ON c.c_customer_sk = cs.cs_ship_customer_sk
    GROUP BY 
        c.c_customer_sk
),
top_customers AS (
    SELECT 
        cs.c_customer_sk,
        cs.total_orders,
        cs.total_spent,
        cs.avg_order_value,
        RANK() OVER (ORDER BY cs.total_spent DESC) AS customer_rank
    FROM 
        customer_summary cs
)
SELECT 
    c.c_customer_id, 
    ca.ca_city,
    ca.ca_state,
    cu.avg_order_value,
    rs.total_quantity,
    rs.total_net_profit
FROM 
    top_customers cu
JOIN 
    customer c ON cu.c_customer_sk = c.c_customer_sk
LEFT JOIN 
    customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
JOIN 
    ranked_sales rs ON c.c_customer_sk = rs.cs_item_sk
WHERE 
    cu.customer_rank <= 10
    AND ca.ca_state IS NOT NULL
    AND rs.total_quantity > 100
ORDER BY 
    cu.total_spent DESC, 
    rs.total_net_profit DESC;
