WITH SalesData AS (
    SELECT 
        ss_store_sk,
        SUM(ss_quantity) AS total_quantity,
        SUM(ss_sales_price) AS total_sales,
        AVG(ss_sales_price) AS avg_sales_price,
        COUNT(DISTINCT ss_ticket_number) AS total_transactions
    FROM 
        store_sales
    WHERE 
        ss_sold_date_sk BETWEEN 2451545 AND 2451549 
    GROUP BY 
        ss_store_sk
), 
CustomerData AS (
    SELECT 
        c.c_customer_sk,
        COUNT(DISTINCT sr_ticket_number) AS total_returns,
        SUM(sr_return_amt) AS total_return_amount,
        AVG(sr_return_quantity) AS avg_return_quantity
    FROM 
        customer c
    LEFT JOIN 
        store_returns sr ON c.c_customer_sk = sr.sr_customer_sk
    GROUP BY 
        c.c_customer_sk
)
SELECT 
    w.w_warehouse_name,
    s.s_store_name,
    sd.total_quantity,
    sd.total_sales,
    sd.avg_sales_price,
    cd.total_returns,
    cd.total_return_amount,
    cd.avg_return_quantity
FROM 
    SalesData sd
JOIN 
    store s ON sd.ss_store_sk = s.s_store_sk
JOIN 
    warehouse w ON s.s_store_sk = w.w_warehouse_sk
LEFT JOIN 
    CustomerData cd ON sd.ss_store_sk = cd.c_customer_sk
ORDER BY 
    total_sales DESC
LIMIT 100;