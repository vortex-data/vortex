
SELECT 
    c.c_customer_id, 
    c.c_first_name, 
    c.c_last_name, 
    SUM(ws.ws_ext_sales_price) AS total_sales, 
    SUM(ws.ws_ext_tax) AS total_tax,
    MAX(d.d_date) AS last_purchase_date,
    COUNT(DISTINCT ws.ws_order_number) AS total_orders,
    ce.cc_call_center_id,
    COUNT(DISTINCT sr.sr_ticket_number) AS total_returns,
    AVG(inv.inv_quantity_on_hand) AS avg_inventory,
    p.p_promo_name
FROM 
    customer AS c
JOIN 
    web_sales AS ws ON c.c_customer_sk = ws.ws_bill_customer_sk
JOIN 
    call_center AS ce ON ws.ws_ship_customer_sk = ce.cc_call_center_sk
JOIN 
    date_dim AS d ON ws.ws_sold_date_sk = d.d_date_sk
LEFT JOIN 
    store_returns AS sr ON c.c_customer_sk = sr.sr_customer_sk 
LEFT JOIN 
    inventory AS inv ON ws.ws_item_sk = inv.inv_item_sk AND inv.inv_warehouse_sk = ws.ws_warehouse_sk
LEFT JOIN 
    promotion AS p ON ws.ws_promo_sk = p.p_promo_sk
WHERE 
    d.d_year = 2023 
    AND c.c_current_cdemo_sk IS NOT NULL
GROUP BY 
    c.c_customer_id, c.c_first_name, c.c_last_name, ce.cc_call_center_id, p.p_promo_name
ORDER BY 
    total_sales DESC 
LIMIT 100;
