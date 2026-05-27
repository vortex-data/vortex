
WITH AddressInfo AS (
    SELECT
        ca_address_sk,
        CONCAT(ca_street_number, ' ', ca_street_name, ' ', ca_street_type) AS full_address,
        ca_city,
        ca_state,
        ca_zip
    FROM
        customer_address
),
DemoInfo AS (
    SELECT
        cd_demo_sk,
        cd_gender,
        CASE
            WHEN cd_marital_status = 'M' THEN 'Married'
            WHEN cd_marital_status = 'S' THEN 'Single'
            ELSE 'Other'
        END AS marital_status,
        cd_education_status,
        cd_purchase_estimate
    FROM
        customer_demographics
),
SalesData AS (
    SELECT
        ws_bill_customer_sk AS customer_sk,
        SUM(ws_sales_price) AS total_sales,
        COUNT(ws_order_number) AS order_count
    FROM
        web_sales
    GROUP BY
        ws_bill_customer_sk
)
SELECT
    c.c_customer_id,
    c.c_first_name,
    c.c_last_name,
    a.full_address,
    d.marital_status,
    d.cd_gender,
    s.total_sales,
    s.order_count
FROM
    customer c
JOIN
    AddressInfo a ON c.c_current_addr_sk = a.ca_address_sk
JOIN
    DemoInfo d ON c.c_current_cdemo_sk = d.cd_demo_sk
LEFT JOIN
    SalesData s ON c.c_customer_sk = s.customer_sk
WHERE
    (a.ca_state = 'CA' AND s.total_sales > 1000)
    OR (d.marital_status = 'Married' AND d.cd_gender = 'F')
ORDER BY
    total_sales DESC NULLS LAST;
