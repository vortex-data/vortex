
SELECT
    CONCAT(c.c_first_name, ' ', c.c_last_name) AS full_name,
    ca.ca_city,
    ca.ca_state,
    ca.ca_zip,
    SUM(ss.ss_ext_sales_price) AS total_sales,
    COUNT(DISTINCT ss.ss_ticket_number) AS number_of_purchases,
    CASE
        WHEN cd.cd_gender = 'M' THEN 'Male'
        WHEN cd.cd_gender = 'F' THEN 'Female'
        ELSE 'Other'
    END AS gender,
    cd.cd_marital_status,
    cd.cd_education_status,
    CASE 
        WHEN cd.cd_purchase_estimate > 5000 THEN 'High Spender'
        WHEN cd.cd_purchase_estimate BETWEEN 1000 AND 5000 THEN 'Medium Spender'
        ELSE 'Low Spender'
    END AS spending_category
FROM 
    customer c 
JOIN 
    customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
JOIN 
    customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
JOIN 
    store_sales ss ON c.c_customer_sk = ss.ss_customer_sk
JOIN 
    date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
WHERE 
    d.d_date BETWEEN '2023-01-01' AND '2023-12-31'
GROUP BY 
    c.c_first_name, c.c_last_name, ca.ca_city, ca.ca_state, ca.ca_zip, 
    cd.cd_gender, cd.cd_marital_status, cd.cd_education_status, 
    cd.cd_purchase_estimate
ORDER BY 
    total_sales DESC
LIMIT 100;
