
WITH Address_Concat AS (
    SELECT 
        ca_address_sk, 
        CONCAT(ca_street_number, ' ', ca_street_name, ' ', ca_street_type, 
               CASE WHEN ca_suite_number IS NOT NULL THEN CONCAT(', Suite ', ca_suite_number) ELSE '' END) AS full_address,
        ca_city, 
        ca_state, 
        ca_zip
    FROM customer_address
),
Customer_Details AS (
    SELECT 
        c.c_customer_sk, 
        CONCAT(c.c_first_name, ' ', c.c_last_name) AS full_name, 
        d.cd_gender,
        d.cd_marital_status,
        d.cd_education_status,
        d.cd_purchase_estimate
    FROM customer c
    JOIN customer_demographics d ON c.c_current_cdemo_sk = d.cd_demo_sk
),
Sales_Info AS (
    SELECT 
        ws_bill_customer_sk,
        SUM(ws_ext_sales_price) AS total_sales,
        COUNT(DISTINCT ws_order_number) AS order_count
    FROM web_sales
    GROUP BY ws_bill_customer_sk
)
SELECT 
    c.full_name, 
    c.cd_gender,
    c.cd_marital_status,
    c.cd_education_status,
    a.full_address,
    a.ca_city,
    a.ca_state,
    a.ca_zip,
    s.total_sales,
    s.order_count
FROM Customer_Details c
JOIN Address_Concat a ON c.c_customer_sk = a.ca_address_sk
LEFT JOIN Sales_Info s ON c.c_customer_sk = s.ws_bill_customer_sk
WHERE c.cd_purchase_estimate > 1000
ORDER BY total_sales DESC, c.full_name;
