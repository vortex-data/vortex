
WITH CustomerSummary AS (
    SELECT 
        CONCAT(c.c_first_name, ' ', c.c_last_name) AS full_name,
        CASE 
            WHEN cd.cd_gender = 'M' THEN 'Male'
            WHEN cd.cd_gender = 'F' THEN 'Female'
            ELSE 'Other'
        END AS gender,
        cd.cd_marital_status AS marital_status,
        cd.cd_education_status AS education_status,
        COUNT(DISTINCT sr.sr_ticket_number) AS total_returns,
        SUM(sr.sr_return_amt) AS total_return_amount,
        COUNT(DISTINCT sr.sr_item_sk) AS distinct_returned_items,
        c.c_customer_sk
    FROM 
        customer c
    JOIN 
        customer_demographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
    LEFT JOIN 
        store_returns sr ON c.c_customer_sk = sr.sr_customer_sk
    GROUP BY 
        c.c_customer_sk, c.c_first_name, c.c_last_name, cd.cd_gender, cd.cd_marital_status, cd.cd_education_status
),
DateDetails AS (
    SELECT 
        d.d_date AS return_date,
        EXTRACT(MONTH FROM d.d_date) AS return_month,
        EXTRACT(YEAR FROM d.d_date) AS return_year,
        CASE 
            WHEN d.d_dow IN (1, 7) THEN 'Weekend'
            ELSE 'Weekday'
        END AS week_day_category,
        d.d_date_sk
    FROM 
        date_dim d
)
SELECT 
    cs.full_name,
    cs.gender,
    cs.marital_status,
    cs.education_status,
    dd.return_month,
    dd.return_year,
    dd.week_day_category,
    cs.total_returns,
    cs.total_return_amount,
    cs.distinct_returned_items
FROM 
    CustomerSummary cs
JOIN 
    store_returns sr ON cs.c_customer_sk = sr.sr_customer_sk
JOIN 
    DateDetails dd ON sr.sr_returned_date_sk = dd.d_date_sk
WHERE 
    cs.total_returns > 0
ORDER BY 
    cs.total_return_amount DESC, cs.full_name ASC
LIMIT 100;
