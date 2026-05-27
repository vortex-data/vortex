
SELECT 
    ca_state, 
    COUNT(DISTINCT c_customer_sk) AS num_customers, 
    SUM(ss_net_profit) AS total_net_profit
FROM 
    customer_address 
JOIN 
    customer ON ca_address_sk = c_current_addr_sk 
JOIN 
    store_sales ON c_customer_sk = ss_customer_sk 
JOIN 
    date_dim ON ss_sold_date_sk = d_date_sk 
WHERE 
    d_year = 2023 
GROUP BY 
    ca_state 
ORDER BY 
    total_net_profit DESC;
