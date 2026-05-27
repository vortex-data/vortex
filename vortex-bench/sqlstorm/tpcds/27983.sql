
WITH AddressData AS (
    SELECT 
        ca_address_sk,
        CONCAT(ca_street_number, ' ', ca_street_name, ' ', ca_street_type) AS full_address,
        TRIM(ca_city) AS city_name,
        ca_state,
        ca_zip
    FROM customer_address
    WHERE ca_country = 'USA'
),
DemographicData AS (
    SELECT 
        cd_demo_sk,
        cd_gender,
        cd_marital_status,
        cd_education_status,
        cd_purchase_estimate,
        cd_credit_rating,
        cd_dep_count
    FROM customer_demographics
    WHERE cd_purchase_estimate > 1000
),
SalesData AS (
    SELECT
        ws_bill_customer_sk,
        SUM(ws_ext_sales_price) AS total_sales
    FROM web_sales
    GROUP BY ws_bill_customer_sk
),
CustomerAddressSales AS (
    SELECT 
        c.c_customer_sk,
        c.c_first_name,
        c.c_last_name,
        a.full_address,
        a.city_name,
        a.ca_state,
        a.ca_zip,
        d.cd_gender,
        d.cd_marital_status,
        d.cd_education_status,
        s.total_sales
    FROM customer c
    JOIN AddressData a ON c.c_current_addr_sk = a.ca_address_sk
    JOIN DemographicData d ON c.c_current_cdemo_sk = d.cd_demo_sk
    LEFT JOIN SalesData s ON c.c_customer_sk = s.ws_bill_customer_sk
)
SELECT 
    city_name,
    ca_state,
    COUNT(*) AS customer_count,
    AVG(total_sales) AS avg_sales,
    SUM(total_sales) AS total_sales
FROM CustomerAddressSales
GROUP BY city_name, ca_state
HAVING COUNT(*) > 10
ORDER BY total_sales DESC;
