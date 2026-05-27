
WITH EnhancedCustomerInfo AS (
    SELECT 
        c.c_customer_sk,
        CONCAT(c.c_salutation, ' ', c.c_first_name, ' ', c.c_last_name) AS full_name,
        ca.ca_city,
        ca.ca_state,
        ca.ca_zip,
        cd.cd_gender,
        cd.cd_marital_status,
        cd.cd_education_status,
        cd.cd_purchase_estimate,
        cd.cd_credit_rating,
        cd.cd_dep_count,
        cd.cd_dep_employed_count,
        cd.cd_dep_college_count
    FROM 
        customer c
    JOIN 
        customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
    JOIN 
        customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
),
SalesSummary AS (
    SELECT 
        ws_bill_customer_sk,
        COUNT(ws_order_number) AS total_orders,
        SUM(ws_net_paid_inc_tax) AS total_revenue
    FROM 
        web_sales
    GROUP BY 
        ws_bill_customer_sk
),
CustomerBenchmarking AS (
    SELECT 
        e.c_customer_sk,
        e.full_name,
        e.ca_city,
        e.ca_state,
        e.ca_zip,
        e.cd_gender,
        e.cd_marital_status,
        e.cd_education_status,
        e.cd_purchase_estimate,
        e.cd_credit_rating,
        s.total_orders,
        s.total_revenue,
        CASE 
            WHEN s.total_orders IS NULL THEN 'No Orders' 
            ELSE 'Active Customer' 
        END AS customer_status
    FROM 
        EnhancedCustomerInfo e
    LEFT JOIN 
        SalesSummary s ON e.c_customer_sk = s.ws_bill_customer_sk
)
SELECT 
    *,
    CASE 
        WHEN total_revenue > 1000 THEN 'High Value Customer'
        WHEN total_revenue BETWEEN 500 AND 1000 THEN 'Medium Value Customer'
        ELSE 'Low Value Customer'
    END AS customer_value
FROM 
    CustomerBenchmarking
ORDER BY 
    total_revenue DESC, full_name;
