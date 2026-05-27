
WITH AddressDetails AS (
    SELECT 
        ca_address_sk,
        CONCAT(ca_street_number, ' ', ca_street_name, ' ', ca_street_type, 
               CASE WHEN ca_suite_number IS NOT NULL THEN CONCAT(' Suite ', ca_suite_number) ELSE '' END) AS Full_Address,
        ca_city,
        ca_state,
        ca_zip,
        ca_country
    FROM customer_address
),
CustomerAggregates AS (
    SELECT 
        c.c_customer_sk,
        CONCAT(c.c_first_name, ' ', c.c_last_name) AS Full_Name,
        cd.cd_gender,
        cd.cd_marital_status,
        COUNT(hd.hd_income_band_sk) AS Income_Band_Count
    FROM customer c
    JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
    LEFT JOIN household_demographics hd ON hd.hd_demo_sk = c.c_current_hdemo_sk
    GROUP BY c.c_customer_sk, c.c_first_name, c.c_last_name, cd.cd_gender, cd.cd_marital_status
),
DateStatistics AS (
    SELECT 
        d.d_year,
        COUNT(ws.ws_order_number) AS Total_Orders,
        SUM(ws.ws_ext_sales_price) AS Total_Sales
    FROM web_sales ws
    JOIN date_dim d ON ws.ws_sold_date_sk = d.d_date_sk
    GROUP BY d.d_year
)
SELECT 
    ca.Full_Address,
    ca.ca_city,
    ca.ca_state,
    ca.ca_zip,
    ca.ca_country,
    cu.Full_Name,
    cu.cd_gender,
    cu.cd_marital_status,
    da.d_year,
    da.Total_Orders,
    da.Total_Sales,
    CASE 
        WHEN da.Total_Sales > 100000 THEN 'High Revenue'
        WHEN da.Total_Sales BETWEEN 50000 AND 100000 THEN 'Moderate Revenue'
        ELSE 'Low Revenue'
    END AS Revenue_Category
FROM AddressDetails ca
JOIN CustomerAggregates cu ON cu.c_customer_sk = ca.ca_address_sk
JOIN DateStatistics da ON da.d_year = EXTRACT(YEAR FROM DATE '2002-10-01')  
WHERE ca.ca_state = 'CA'  
ORDER BY da.Total_Sales DESC, cu.Full_Name;
