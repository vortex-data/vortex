
WITH customer_info AS (
    SELECT c.c_customer_sk, 
           c.c_customer_id, 
           cd.cd_gender,
           cd.cd_marital_status,
           cd.cd_purchase_estimate,
           ca.ca_city,
           ca.ca_state
    FROM customer c
    LEFT JOIN customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
    LEFT JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
),

address_stats AS (
    SELECT ca_state,
           COUNT(DISTINCT c_customer_sk) AS customer_count,
           AVG(cd_purchase_estimate) AS avg_purchase_estimate
    FROM customer_info
    GROUP BY ca_state
),

ranked_customers AS (
    SELECT ci.c_customer_id,
           ci.cd_gender,
           ci.ca_city,
           ci.ca_state,
           RANK() OVER (PARTITION BY ci.ca_state ORDER BY ci.cd_purchase_estimate DESC) AS purchase_rank
    FROM customer_info ci
),

sales_data AS (
    SELECT ws_bill_customer_sk,
           SUM(ws_sales_price) AS total_sales
    FROM web_sales
    GROUP BY ws_bill_customer_sk
),

return_data AS (
    SELECT sr_customer_sk,
           SUM(sr_return_amt_inc_tax) AS total_returns
    FROM store_returns
    GROUP BY sr_customer_sk
),

final_analysis AS (
    SELECT ci.c_customer_id,
           ci.cd_gender,
           ci.ca_city,
           ci.ca_state,
           COALESCE(sd.total_sales, 0) AS total_sales,
           COALESCE(rd.total_returns, 0) AS total_returns,
           (COALESCE(sd.total_sales, 0) - COALESCE(rd.total_returns, 0)) AS net_revenue,
           (SELECT AVG(avg_purchase_estimate) FROM address_stats AS a WHERE a.ca_state = ci.ca_state) AS state_avg_purchase,
           (CASE 
               WHEN (COALESCE(sd.total_sales, 0) - COALESCE(rd.total_returns, 0)) < 0 THEN 'Negative Revenue'
               WHEN (COALESCE(sd.total_sales, 0) - COALESCE(rd.total_returns, 0)) = 0 THEN 'Break Even'
               ELSE 'Positive Revenue'
           END) AS revenue_status
    FROM customer_info ci
    LEFT JOIN sales_data sd ON ci.c_customer_sk = sd.ws_bill_customer_sk
    LEFT JOIN return_data rd ON ci.c_customer_sk = rd.sr_customer_sk
)

SELECT fa.c_customer_id,
       fa.cd_gender,
       fa.ca_city,
       fa.ca_state,
       fa.total_sales,
       fa.total_returns,
       fa.net_revenue,
       fa.state_avg_purchase,
       fa.revenue_status,
       rk.purchase_rank
FROM final_analysis fa
JOIN ranked_customers rk ON fa.c_customer_id = rk.c_customer_id
WHERE rk.purchase_rank <= 10 OR fa.net_revenue < 0
ORDER BY fa.ca_state, fa.net_revenue DESC, rk.purchase_rank;
