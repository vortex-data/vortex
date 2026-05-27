
WITH SalesData AS (
    SELECT 
        ws_item_sk,
        SUM(ws_ext_sales_price) AS total_sales,
        SUM(ws_net_profit) AS total_profit,
        COUNT(DISTINCT ws_order_number) AS order_count
    FROM 
        web_sales
    WHERE 
        ws_sold_date_sk BETWEEN (SELECT MIN(d_date_sk) FROM date_dim WHERE d_year = 2023) AND 
                               (SELECT MAX(d_date_sk) FROM date_dim WHERE d_year = 2023)
    GROUP BY 
        ws_item_sk
),
TopItems AS (
    SELECT 
        i.i_item_sk,
        i.i_item_desc,
        dd.d_year,
        RANK() OVER (PARTITION BY dd.d_year ORDER BY sd.total_sales DESC) AS sales_rank
    FROM 
        Item i
    JOIN 
        SalesData sd ON i.i_item_sk = sd.ws_item_sk
    JOIN 
        date_dim dd ON dd.d_date_sk = sd.ws_item_sk  -- Fixed the join condition
)
SELECT 
    ti.i_item_desc,
    ti.sales_rank,
    sd.total_sales,
    sd.total_profit,
    sd.order_count
FROM 
    TopItems ti
JOIN 
    SalesData sd ON ti.i_item_sk = sd.ws_item_sk
WHERE 
    ti.sales_rank <= 10
ORDER BY 
    sd.total_sales DESC;
