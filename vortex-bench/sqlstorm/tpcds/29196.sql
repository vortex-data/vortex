
WITH CustomerDetails AS (
    SELECT 
        c.c_customer_sk,
        CONCAT(c.c_first_name, ' ', c.c_last_name) AS customer_name,
        addr.ca_city,
        addr.ca_state,
        dem.cd_gender,
        dem.cd_marital_status,
        dem.cd_education_status,
        dem.cd_purchase_estimate
    FROM 
        customer c
    JOIN 
        customer_address addr ON c.c_current_addr_sk = addr.ca_address_sk
    JOIN 
        customer_demographics dem ON c.c_current_cdemo_sk = dem.cd_demo_sk
),
SalesSummary AS (
    SELECT 
        cd.c_customer_sk,
        SUM(ws.ws_sales_price) AS total_sales,
        COUNT(ws.ws_order_number) AS total_orders
    FROM 
        web_sales ws
    JOIN 
        CustomerDetails cd ON ws.ws_bill_customer_sk = cd.c_customer_sk
    GROUP BY 
        cd.c_customer_sk
),
FinalReport AS (
    SELECT 
        cd.customer_name,
        cd.ca_city,
        cd.ca_state,
        cd.cd_gender,
        cd.cd_marital_status,
        cd.cd_education_status,
        ss.total_sales,
        ss.total_orders
    FROM 
        CustomerDetails cd
    LEFT JOIN 
        SalesSummary ss ON cd.c_customer_sk = ss.c_customer_sk
)
SELECT 
    customer_name,
    ca_city,
    ca_state,
    cd_gender,
    cd_marital_status,
    cd_education_status,
    COALESCE(total_sales, 0) AS total_sales,
    COALESCE(total_orders, 0) AS total_orders,
    CASE 
        WHEN COALESCE(total_sales, 0) = 0 THEN 'No Sales'
        WHEN COALESCE(total_sales, 0) < 100 THEN 'Low Sales'
        WHEN COALESCE(total_sales, 0) BETWEEN 100 AND 500 THEN 'Moderate Sales'
        ELSE 'High Sales'
    END AS sales_category
FROM 
    FinalReport
ORDER BY 
    total_sales DESC, customer_name;
