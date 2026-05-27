WITH sales_data AS (
    SELECT 
        ws.ws_item_sk,
        SUM(ws.ws_quantity) AS total_quantity,
        SUM(ws.ws_net_profit) AS total_net_profit,
        DENSE_RANK() OVER (PARTITION BY ws.ws_item_sk ORDER BY SUM(ws.ws_net_profit) DESC) AS profit_rank
    FROM 
        web_sales ws
    JOIN 
        item i ON ws.ws_item_sk = i.i_item_sk
    WHERE 
        ws.ws_sold_date_sk BETWEEN 2400 AND 2470 
    GROUP BY 
        ws.ws_item_sk
),
top_items AS (
    SELECT 
        sd.ws_item_sk,
        sd.total_quantity,
        sd.total_net_profit,
        i.i_item_desc,
        ROW_NUMBER() OVER (ORDER BY sd.total_net_profit DESC) AS row_num
    FROM 
        sales_data sd
    JOIN 
        item i ON sd.ws_item_sk = i.i_item_sk
    WHERE 
        sd.profit_rank <= 10
    ORDER BY 
        sd.total_net_profit DESC
)
SELECT 
    ti.row_num,
    ti.i_item_desc,
    ti.total_quantity,
    COALESCE(pr.p_promo_name, 'No Promotion') AS promotion_name,
    ti.total_net_profit
FROM 
    top_items ti
LEFT JOIN 
    promotion pr ON ti.ws_item_sk = pr.p_item_sk AND pr.p_start_date_sk <= 2470 AND pr.p_end_date_sk >= 2400
ORDER BY 
    ti.row_num;