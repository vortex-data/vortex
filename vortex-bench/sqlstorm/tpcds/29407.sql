
WITH AddressInfo AS (
    SELECT 
        ca_city,
        COUNT(DISTINCT ca_zip) AS unique_zip_count,
        COUNT(*) AS total_addresses,
        AVG(LENGTH(ca_street_name) + LENGTH(ca_street_type)) AS avg_street_length
    FROM 
        customer_address
    GROUP BY 
        ca_city
),
CustomerInfo AS (
    SELECT 
        cd_gender,
        cd_marital_status,
        COUNT(DISTINCT c_customer_id) AS customer_count,
        AVG(cd_purchase_estimate) AS avg_purchase_estimate
    FROM 
        customer_demographics
    JOIN 
        customer ON cd_demo_sk = c_current_cdemo_sk
    GROUP BY 
        cd_gender, cd_marital_status
),
InventorySnapshot AS (
    SELECT 
        inv_warehouse_sk,
        COUNT(DISTINCT inv_item_sk) AS unique_items,
        SUM(inv_quantity_on_hand) AS total_quantity
    FROM 
        inventory
    GROUP BY 
        inv_warehouse_sk
),
SalesSummary AS (
    SELECT 
        ws_ship_mode_sk,
        SUM(ws_quantity) AS total_quantity_sold,
        SUM(ws_sales_price * ws_quantity) AS total_sales_value
    FROM 
        web_sales
    GROUP BY 
        ws_ship_mode_sk
)
SELECT 
    a.ca_city,
    a.unique_zip_count,
    a.total_addresses,
    a.avg_street_length,
    c.cd_gender,
    c.cd_marital_status,
    c.customer_count,
    c.avg_purchase_estimate,
    i.unique_items,
    i.total_quantity,
    s.total_quantity_sold,
    s.total_sales_value
FROM 
    AddressInfo a
JOIN 
    CustomerInfo c ON a.total_addresses > 100
JOIN 
    InventorySnapshot i ON i.unique_items > 50
JOIN 
    SalesSummary s ON s.total_sales_value > 10000
ORDER BY 
    a.ca_city, c.cd_gender;
