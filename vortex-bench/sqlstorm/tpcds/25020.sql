
WITH Address_Enhanced AS (
    SELECT 
        ca_address_sk,
        TRIM(ca_street_number) || ' ' || 
        UPPER(LEFT(ca_street_name, 1)) || LOWER(SUBSTRING(ca_street_name FROM 2)) || ' ' || 
        UPPER(ca_street_type) AS Full_Address,
        ca_city,
        ca_state,
        ca_zip,
        ca_country
    FROM customer_address
),
Customer_Enhanced AS (
    SELECT 
        c.c_customer_sk,
        CONCAT(UPPER(SUBSTRING(c.c_first_name, 1, 1)), LOWER(SUBSTRING(c.c_first_name FROM 2)), ' ', 
               UPPER(SUBSTRING(c.c_last_name, 1, 1)), LOWER(SUBSTRING(c.c_last_name FROM 2))) AS Full_Name,
        cd.cd_gender,
        cd.cd_marital_status,
        cd.cd_education_status,
        cd.cd_purchase_estimate,
        ae.Full_Address,
        ae.ca_city,
        ae.ca_state,
        ae.ca_zip,
        ae.ca_country
    FROM customer c
    JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
    JOIN Address_Enhanced ae ON c.c_current_addr_sk = ae.ca_address_sk
),
Sales_Enhanced AS (
    SELECT 
        ws.ws_item_sk,
        SUM(ws.ws_quantity) AS Total_Quantity,
        SUM(ws.ws_net_profit) AS Total_Profit
    FROM web_sales ws
    GROUP BY ws.ws_item_sk
)
SELECT 
    ce.Full_Name,
    ce.cd_gender,
    ce.cd_marital_status,
    ce.cd_education_status,
    ce.cd_purchase_estimate,
    se.Total_Quantity,
    se.Total_Profit,
    ce.Full_Address,
    ce.ca_city,
    ce.ca_state,
    ce.ca_zip,
    ce.ca_country
FROM Customer_Enhanced ce
JOIN Sales_Enhanced se ON ce.c_customer_sk = se.ws_item_sk
WHERE UPPER(ce.ca_state) = 'CA' 
AND ce.cd_purchase_estimate > 500
ORDER BY se.Total_Profit DESC
LIMIT 100;
