
WITH CustomerInfo AS (
    SELECT 
        c.c_customer_sk,
        CONCAT(c.c_first_name, ' ', c.c_last_name) AS full_name,
        cd.cd_gender,
        cd.cd_marital_status,
        cd.cd_education_status,
        cd.cd_purchase_estimate,
        ca.ca_city,
        ca.ca_state,
        ca.ca_country
    FROM 
        customer c
    JOIN 
        customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
    JOIN 
        customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
),
SalesInfo AS (
    SELECT 
        ws_bill_customer_sk,
        SUM(ws_net_paid) AS total_sales
    FROM 
        web_sales
    GROUP BY 
        ws_bill_customer_sk
),
ReturnsInfo AS (
    SELECT 
        wr_returning_customer_sk,
        SUM(wr_return_amt) AS total_returns
    FROM 
        web_returns
    GROUP BY 
        wr_returning_customer_sk
),
CombinedInfo AS (
    SELECT 
        ci.full_name,
        ci.ca_city,
        ci.ca_state,
        ci.ca_country,
        COALESCE(si.total_sales, 0) AS total_sales,
        COALESCE(ri.total_returns, 0) AS total_returns,
        COALESCE(si.total_sales, 0) - COALESCE(ri.total_returns, 0) AS net_profit
    FROM 
        CustomerInfo ci
    LEFT JOIN 
        SalesInfo si ON ci.c_customer_sk = si.ws_bill_customer_sk
    LEFT JOIN 
        ReturnsInfo ri ON ci.c_customer_sk = ri.wr_returning_customer_sk
)
SELECT 
    full_name,
    ca_city,
    ca_state,
    ca_country,
    total_sales,
    total_returns,
    net_profit,
    RANK() OVER (ORDER BY net_profit DESC) AS profit_rank
FROM 
    CombinedInfo
WHERE 
    net_profit > 0
ORDER BY 
    profit_rank
LIMIT 10;
