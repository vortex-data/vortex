
WITH address_summary AS (
    SELECT
        ca_city AS city,
        ca_state AS state,
        COUNT(DISTINCT ca_address_sk) AS total_addresses,
        SUM(LENGTH(ca_street_name) + LENGTH(ca_street_number) + LENGTH(ca_street_type)) AS total_characters,
        AVG(LENGTH(ca_street_name) + LENGTH(ca_street_number) + LENGTH(ca_street_type)) AS avg_address_length
    FROM
        customer_address
    GROUP BY
        ca_city, ca_state
),
customer_summary AS (
    SELECT
        cd_gender,
        COUNT(DISTINCT c_customer_sk) AS total_customers,
        SUM(cd_dep_count) AS total_dependents,
        AVG(cd_purchase_estimate) AS avg_purchase_estimate
    FROM
        customer
    JOIN
        customer_demographics ON c_current_cdemo_sk = cd_demo_sk
    GROUP BY
        cd_gender
),
sales_summary AS (
    SELECT
        ws_bill_customer_sk,
        SUM(ws_net_profit) AS total_profit,
        COUNT(ws_order_number) AS total_orders,
        SUM(ws_quantity) AS total_items_sold
    FROM
        web_sales
    GROUP BY
        ws_bill_customer_sk
)
SELECT
    ca.city AS city,
    ca.state AS state,
    ca.total_addresses,
    ca.total_characters,
    ca.avg_address_length,
    cu.cd_gender,
    cu.total_customers,
    cu.total_dependents,
    cu.avg_purchase_estimate,
    ss.total_profit,
    ss.total_orders,
    ss.total_items_sold
FROM
    address_summary ca
JOIN
    customer_summary cu ON ca.total_addresses > 100  
JOIN
    sales_summary ss ON ss.ws_bill_customer_sk = cu.total_customers
WHERE
    ca.total_characters > 1000  
ORDER BY
    ca.city, cu.cd_gender;
