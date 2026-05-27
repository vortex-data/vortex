
WITH RECURSIVE sales_data AS (
    SELECT
        ws_order_number,
        ws_item_sk,
        ws_sold_date_sk,
        ws_quantity,
        ws_sales_price,
        ROW_NUMBER() OVER (PARTITION BY ws_item_sk ORDER BY ws_sold_date_sk DESC) AS rn
    FROM 
        web_sales
    WHERE 
        ws_sold_date_sk >= 20210101
),
inventory_data AS (
    SELECT 
        inv_date_sk,
        inv_item_sk,
        SUM(inv_quantity_on_hand) AS total_quantity
    FROM 
        inventory
    WHERE 
        inv_date_sk BETWEEN 20210101 AND 20220331
    GROUP BY 
        inv_date_sk, inv_item_sk
),
customer_segments AS (
    SELECT 
        cd_demo_sk,
        COUNT(DISTINCT c_customer_sk) AS customer_count,
        MAX(cd_purchase_estimate) AS max_purchase_estimate
    FROM 
        customer_demographics
    JOIN 
        customer ON c_current_cdemo_sk = cd_demo_sk
    GROUP BY 
        cd_demo_sk
)
SELECT 
    c.c_first_name,
    c.c_last_name,
    sa.ws_order_number,
    sa.ws_item_sk,
    sa.ws_quantity,
    sa.ws_sales_price,
    case 
        when (sa.ws_quantity > id.total_quantity) then 'Exceeds Inventory' 
        else 'Within Inventory' 
    end as inventory_status,
    cs.customer_count AS segment_customer_count,
    cs.max_purchase_estimate
FROM 
    customer c
JOIN 
    sales_data sa ON c.c_customer_sk = sa.ws_order_number
LEFT JOIN 
    inventory_data id ON sa.ws_item_sk = id.inv_item_sk
LEFT JOIN 
    customer_segments cs ON c.c_current_cdemo_sk = cs.cd_demo_sk
WHERE 
    sa.rn = 1 
    AND c.c_birth_year IS NOT NULL
ORDER BY 
    c.c_last_name, c.c_first_name;
