
WITH AddressComponents AS (
    SELECT 
        ca_address_sk,
        CONCAT(ca_street_number, ' ', ca_street_name, ' ', ca_street_type) AS full_address,
        ca_city,
        ca_state,
        ca_zip
    FROM 
        customer_address
),
CustomerDetails AS (
    SELECT 
        c.c_customer_sk,
        c.c_first_name,
        c.c_last_name,
        cd.cd_gender,
        cd.cd_marital_status,
        cd.cd_credit_rating,
        ac.full_address,
        ac.ca_city,
        ac.ca_state,
        ac.ca_zip
    FROM 
        customer c
    JOIN 
        customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
    JOIN 
        AddressComponents ac ON c.c_current_addr_sk = ac.ca_address_sk
),
SalesDetails AS (
    SELECT 
        ws.ws_order_number,
        ws.ws_quantity,
        ws.ws_ext_sales_price,
        ws.ws_net_paid,
        cd.c_customer_sk
    FROM 
        web_sales ws
    JOIN 
        CustomerDetails cd ON ws.ws_bill_customer_sk = cd.c_customer_sk
)
SELECT 
    cd.c_first_name,
    cd.c_last_name,
    cd.ca_city,
    cd.ca_state,
    COUNT(sd.ws_order_number) AS total_orders,
    SUM(sd.ws_quantity) AS total_quantity,
    SUM(sd.ws_ext_sales_price) AS total_sales,
    AVG(sd.ws_net_paid) AS avg_net_paid
FROM 
    CustomerDetails cd
LEFT JOIN 
    SalesDetails sd ON cd.c_customer_sk = sd.c_customer_sk
GROUP BY 
    cd.c_first_name, cd.c_last_name, cd.ca_city, cd.ca_state
ORDER BY 
    total_sales DESC
LIMIT 50;
