
WITH AddressDetails AS (
    SELECT 
        ca_address_sk,
        CONCAT(ca_street_number, ' ', ca_street_name, ' ', ca_street_type) AS full_address,
        ca_city,
        ca_state,
        ca_zip,
        ca_country
    FROM 
        customer_address
),
CustomerInfo AS (
    SELECT 
        c_customer_sk,
        CONCAT(c_first_name, ' ', c_last_name) AS full_name,
        cd_gender,
        cd_marital_status,
        cd_education_status,
        cd_purchase_estimate,
        cd_credit_rating
    FROM 
        customer c
    JOIN 
        customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
),
SalesSummary AS (
    SELECT 
        ws_bill_customer_sk AS customer_id,
        COUNT(ws_order_number) AS total_orders,
        SUM(ws_ext_sales_price) AS total_revenue,
        SUM(ws_ext_discount_amt) AS total_discount
    FROM 
        web_sales
    GROUP BY 
        ws_bill_customer_sk
),
CombinedInfo AS (
    SELECT 
        ci.full_name,
        ad.full_address,
        ad.ca_city,
        ad.ca_state,
        ad.ca_zip,
        ad.ca_country,
        si.total_orders,
        si.total_revenue,
        si.total_discount
    FROM 
        CustomerInfo ci
    JOIN 
        AddressDetails ad ON ci.c_customer_sk = ad.ca_address_sk
    JOIN 
        SalesSummary si ON ci.c_customer_sk = si.customer_id
)
SELECT 
    full_name,
    full_address,
    ca_city,
    ca_state,
    ca_zip,
    ca_country,
    total_orders,
    total_revenue,
    total_discount,
    CASE 
        WHEN total_revenue > 1000 THEN 'High Value Customer' 
        WHEN total_revenue BETWEEN 500 AND 1000 THEN 'Medium Value Customer' 
        ELSE 'Low Value Customer' 
    END AS customer_segment
FROM 
    CombinedInfo
ORDER BY 
    total_revenue DESC;
