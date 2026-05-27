
WITH RankedReturns AS (
    SELECT 
        sr_returned_date_sk, 
        SUM(sr_return_quantity) AS total_returned_quantity,
        ROW_NUMBER() OVER (PARTITION BY sr_item_sk ORDER BY SUM(sr_return_quantity) DESC) AS rk
    FROM 
        store_returns
    GROUP BY 
        sr_item_sk, sr_returned_date_sk
), 
CustomerReturns AS (
    SELECT 
        sr_customer_sk, 
        COUNT(DISTINCT sr_ticket_number) AS return_count,
        MAX(CASE WHEN sr_reason_sk IS NULL THEN 1 ELSE 0 END) AS null_reason_return
    FROM 
        store_returns
    WHERE 
        sr_return_quantity > 0
    GROUP BY 
        sr_customer_sk
),
CustomerDemographics AS (
    SELECT 
        cd_gender,
        COUNT(DISTINCT c_customer_sk) AS customer_count,
        MAX(cd_purchase_estimate) AS max_purchase_estimate
    FROM 
        customer_demographics 
    INNER JOIN 
        customer ON customer.c_current_cdemo_sk = customer_demographics.cd_demo_sk
    GROUP BY 
        cd_gender
),
DateDynamics AS (
    SELECT 
        d_year,
        COUNT(ws_order_number) AS total_orders,
        SUM(ws_net_profit) AS total_profit,
        AVG(ws_net_profit) AS avg_profit_per_order
    FROM 
        web_sales
    JOIN 
        date_dim ON ws_sold_date_sk = d_date_sk
    GROUP BY 
        d_year
)
SELECT 
    cd.cd_gender, 
    cd.customer_count,
    cd.max_purchase_estimate,
    dd.d_year,
    dd.total_orders,
    dd.total_profit,
    dd.avg_profit_per_order,
    COALESCE(cr.return_count, 0) AS total_customer_returns,
    COALESCE(cr.null_reason_return, 0) AS returns_with_null_reason
FROM 
    CustomerDemographics cd
JOIN 
    DateDynamics dd ON dd.total_orders > 1000
LEFT JOIN 
    CustomerReturns cr ON cr.sr_customer_sk = cd.customer_count
WHERE 
    cd.max_purchase_estimate > 1000
    AND (cd_gender = 'M' OR cd_gender = 'F')
ORDER BY 
    cd_gender DESC, 
    total_profit DESC
LIMIT 100;
