
WITH RECURSIVE SalesCTE AS (
    SELECT 
        ss_customer_sk,
        SUM(ss_net_paid) AS total_sales,
        COUNT(ss_ticket_number) AS sales_count,
        RANK() OVER (ORDER BY SUM(ss_net_paid) DESC) AS sales_rank
    FROM 
        store_sales
    WHERE 
        ss_sold_date_sk >= (SELECT MIN(d_date_sk) FROM date_dim WHERE d_year = 2023)
        AND ss_sold_date_sk <= (SELECT MAX(d_date_sk) FROM date_dim WHERE d_year = 2023)
    GROUP BY 
        ss_customer_sk
),
AddressCTE AS (
    SELECT 
        ca_address_sk,
        ca_city,
        ca_state,
        ca_country,
        ROW_NUMBER() OVER (PARTITION BY ca_state ORDER BY ca_city) AS city_rank
    FROM 
        customer_address
),
HighValueCustomers AS (
    SELECT 
        c.c_customer_sk,
        c.c_first_name,
        c.c_last_name,
        cd.cd_gender,
        sd.total_sales,
        sd.sales_count,
        a.ca_city,
        a.ca_state,
        a.ca_country
    FROM 
        customer c
    JOIN 
        customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
    JOIN 
        SalesCTE sd ON c.c_customer_sk = sd.ss_customer_sk
    LEFT JOIN 
        AddressCTE a ON c.c_current_addr_sk = a.ca_address_sk
    WHERE 
        sd.total_sales > (SELECT AVG(total_sales) FROM SalesCTE)
        AND cd.cd_marital_status = 'M'
        AND cd.cd_gender = 'F'
),
TopCities AS (
    SELECT 
        ca_city, 
        COUNT(DISTINCT c_customer_sk) AS customer_count,
        SUM(total_sales) AS city_sales
    FROM 
        HighValueCustomers
    GROUP BY 
        ca_city
    HAVING 
        COUNT(DISTINCT c_customer_sk) > 10
)
SELECT 
    tc.ca_city,
    tc.customer_count,
    tc.city_sales,
    CASE 
        WHEN tc.city_sales > 10000 THEN 'High'
        WHEN tc.city_sales BETWEEN 1000 AND 10000 THEN 'Medium'
        ELSE 'Low'
    END AS sales_category
FROM 
    TopCities tc
ORDER BY 
    tc.city_sales DESC
LIMIT 10;
