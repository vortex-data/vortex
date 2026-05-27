
WITH AddressDetails AS (
    SELECT 
        ca.ca_address_sk,
        CONCAT(ca.ca_street_number, ' ', ca.ca_street_name, ' ', ca.ca_street_type, 
               CASE WHEN ca.ca_suite_number IS NOT NULL THEN CONCAT(' Ste ', ca.ca_suite_number) ELSE '' END) AS full_address,
        CONCAT(ca.ca_city, ', ', ca.ca_state, ' ', ca.ca_zip) AS city_state_zip,
        ca.ca_country
    FROM 
        customer_address ca
),
CustomerDemographics AS (
    SELECT 
        cd.cd_demo_sk,
        cd.cd_gender,
        cd.cd_marital_status,
        cd.cd_education_status,
        cd.cd_purchase_estimate,
        cd.cd_credit_rating
    FROM 
        customer_demographics cd
),
CustomerDetails AS (
    SELECT 
        c.c_customer_sk,
        c.c_first_name,
        c.c_last_name,
        c.c_email_address,
        ad.full_address,
        ad.city_state_zip,
        ad.ca_country
    FROM 
        customer c
    JOIN AddressDetails ad ON c.c_current_addr_sk = ad.ca_address_sk
),
SalesOverview AS (
    SELECT 
        ws.ws_item_sk,
        SUM(ws.ws_quantity) AS total_quantity,
        SUM(ws.ws_net_paid) AS total_sales,
        COUNT(DISTINCT ws.ws_order_number) AS total_orders
    FROM 
        web_sales ws
    GROUP BY 
        ws.ws_item_sk
)
SELECT 
    cd.c_first_name,
    cd.c_last_name,
    cd.c_email_address,
    cd.city_state_zip,
    cd.ca_country,
    so.total_quantity,
    so.total_sales,
    so.total_orders
FROM 
    CustomerDetails cd
JOIN SalesOverview so ON cd.c_customer_sk = so.ws_item_sk
WHERE 
    cd.c_email_address LIKE '%@example.com'
ORDER BY 
    so.total_sales DESC 
LIMIT 100;
