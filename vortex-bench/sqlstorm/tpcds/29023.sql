
WITH CustomerInfo AS (
    SELECT 
        c.c_customer_sk,
        CONCAT(c.c_salutation, ' ', c.c_first_name, ' ', c.c_last_name) AS full_name,
        cd.cd_gender,
        cd.cd_marital_status,
        cd.cd_education_status,
        ca.ca_city,
        ca.ca_state
    FROM 
        customer c
    JOIN 
        customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
    JOIN 
        customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
),
SalesInfo AS (
    SELECT 
        SUM(ss.ss_quantity) AS total_quantity,
        SUM(ss.ss_sales_price) AS total_sales,
        ss.ss_customer_sk
    FROM 
        store_sales ss
    GROUP BY 
        ss.ss_customer_sk
),
InfoWithSales AS (
    SELECT 
        ci.full_name,
        ci.cd_gender,
        ci.cd_marital_status,
        ci.cd_education_status,
        ci.ca_city,
        ci.ca_state,
        COALESCE(si.total_quantity, 0) AS total_quantity,
        COALESCE(si.total_sales, 0) AS total_sales
    FROM 
        CustomerInfo ci
    LEFT JOIN 
        SalesInfo si ON ci.c_customer_sk = si.ss_customer_sk
)
SELECT 
    cd_gender,
    cd_marital_status,
    COUNT(*) AS customer_count,
    AVG(total_quantity) AS avg_quantity,
    AVG(total_sales) AS avg_sales,
    ca_state
FROM 
    InfoWithSales
GROUP BY 
    cd_gender, 
    cd_marital_status, 
    ca_state
ORDER BY 
    ca_state, 
    cd_gender, 
    cd_marital_status;
