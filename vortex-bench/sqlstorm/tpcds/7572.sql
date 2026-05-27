
WITH CustomerReturns AS (
    SELECT 
        wr_returning_customer_sk AS customer_sk,
        SUM(wr_return_quantity) AS total_returned_items,
        SUM(wr_return_amt) AS total_return_amount,
        COUNT(DISTINCT wr_order_number) AS return_count
    FROM 
        web_returns
    GROUP BY 
        wr_returning_customer_sk
),
CustomerDemographics AS (
    SELECT 
        c.c_customer_sk,
        cd.cd_gender,
        cd.cd_marital_status,
        cd.cd_education_status,
        cd.cd_purchase_estimate
    FROM 
        customer c
    JOIN 
        customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
),
ReturnStatistics AS (
    SELECT 
        cd.c_customer_sk,
        cd.cd_gender,
        cd.cd_marital_status,
        COUNT(CASE WHEN cr.return_count > 1 THEN 1 END) AS repeat_returns,
        AVG(cr.total_return_amount) AS avg_return_amount,
        SUM(cr.total_returned_items) AS total_returned
    FROM 
        CustomerDemographics cd
    LEFT JOIN 
        CustomerReturns cr ON cd.c_customer_sk = cr.customer_sk
    GROUP BY 
        cd.c_customer_sk, cd.cd_gender, cd.cd_marital_status
),
TopReturners AS (
    SELECT 
        r.cd_gender,
        r.cd_marital_status,
        COUNT(*) AS customer_count,
        SUM(r.total_returned) AS total_items_returned
    FROM 
        ReturnStatistics r
    GROUP BY 
        r.cd_gender, r.cd_marital_status
    ORDER BY 
        total_items_returned DESC
)
SELECT 
    t.cd_gender,
    t.cd_marital_status,
    t.customer_count,
    t.total_items_returned
FROM 
    TopReturners t
WHERE 
    t.customer_count > 10
ORDER BY 
    t.total_items_returned DESC;
