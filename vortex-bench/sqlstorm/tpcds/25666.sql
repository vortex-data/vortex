
WITH AddressStats AS (
    SELECT 
        ca_state,
        COUNT(*) AS total_addresses,
        AVG(LENGTH(ca_street_name)) AS avg_street_name_length,
        SUM(CASE WHEN ca_street_type IS NOT NULL THEN 1 ELSE 0 END) AS street_type_count
    FROM 
        customer_address
    GROUP BY 
        ca_state
),
CustomerStats AS (
    SELECT 
        cd_gender,
        COUNT(*) AS total_customers,
        AVG(cd_purchase_estimate) AS avg_purchase_estimate,
        MAX(cd_dep_count) AS max_dependents
    FROM 
        customer_demographics
    GROUP BY 
        cd_gender
),
SalesStats AS (
    SELECT 
        sm.sm_type,
        COUNT(ws.ws_order_number) AS total_sales,
        SUM(ws.ws_sales_price) AS total_revenue,
        AVG(ws.ws_net_profit) AS avg_net_profit
    FROM 
        web_sales ws
    JOIN 
        ship_mode sm ON ws.ws_ship_mode_sk = sm.sm_ship_mode_sk
    GROUP BY 
        sm.sm_type
)
SELECT 
    A.ca_state, 
    A.total_addresses, 
    A.avg_street_name_length, 
    A.street_type_count,
    C.cd_gender,
    C.total_customers, 
    C.avg_purchase_estimate, 
    C.max_dependents,
    S.sm_type,
    S.total_sales,
    S.total_revenue,
    S.avg_net_profit
FROM 
    AddressStats A
JOIN 
    CustomerStats C ON C.total_customers > 1000
JOIN 
    SalesStats S ON S.total_sales > 50
ORDER BY 
    A.total_addresses DESC, 
    C.total_customers DESC, 
    S.total_revenue DESC;
