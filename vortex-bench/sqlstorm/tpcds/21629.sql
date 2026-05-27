
WITH ranked_customers AS (
    SELECT 
        c.c_customer_sk,
        c.c_first_name,
        c.c_last_name,
        cd.cd_gender,
        cd.cd_marital_status,
        ROW_NUMBER() OVER (PARTITION BY cd.cd_gender ORDER BY cd.cd_purchase_estimate DESC) AS rank
    FROM 
        customer c
    JOIN 
        customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
),
top_customers AS (
    SELECT * FROM ranked_customers WHERE rank <= 10
),
sales_info AS (
    SELECT 
        ws.ws_item_sk,
        SUM(ws.ws_sales_price) AS total_sales,
        COUNT(ws.ws_order_number) AS total_orders,
        AVG(ws.ws_net_profit) AS avg_profit
    FROM 
        web_sales ws
    JOIN 
        top_customers tc ON ws.ws_bill_customer_sk = tc.c_customer_sk
    GROUP BY 
        ws.ws_item_sk
),
inventory_data AS (
    SELECT 
        inv.inv_item_sk,
        inv.inv_quantity_on_hand,
        CASE 
            WHEN inv.inv_quantity_on_hand IS NULL THEN 'Out of Stock'
            ELSE 'In Stock'
        END AS stock_status
    FROM 
        inventory inv
),
final_result AS (
    SELECT 
        si.ws_item_sk,
        si.total_sales,
        si.total_orders,
        si.avg_profit,
        COALESCE(id.inv_quantity_on_hand, 0) AS available_quantity,
        id.stock_status
    FROM 
        sales_info si
    LEFT JOIN 
        inventory_data id ON si.ws_item_sk = id.inv_item_sk
)
SELECT 
    fr.ws_item_sk,
    fr.total_sales,
    fr.total_orders,
    fr.avg_profit,
    fr.available_quantity,
    fr.stock_status,
    CASE 
        WHEN fr.avg_profit > 100 THEN 'High Profit'
        WHEN fr.avg_profit IS NULL THEN 'No Profit Data'
        ELSE 'Moderate Profit'
    END AS profit_category
FROM 
    final_result fr
WHERE 
    fr.total_sales > 1000 
    OR fr.total_orders > 50
ORDER BY 
    fr.total_sales DESC, 
    fr.total_orders ASC;
