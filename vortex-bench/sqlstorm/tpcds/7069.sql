
WITH SalesSummary AS (
    SELECT 
        ws_bill_cdemo_sk AS CustomerDemoSK,
        SUM(ws_ext_sales_price) AS TotalSales,
        COUNT(DISTINCT ws_order_number) AS TotalOrders,
        AVG(ws_sales_price) AS AvgPrice,
        MAX(ws_sales_price) AS MaxPrice,
        MIN(ws_sales_price) AS MinPrice
    FROM 
        web_sales
    WHERE 
        ws_sold_date_sk BETWEEN (SELECT d_date_sk FROM date_dim WHERE d_date = '2023-10-01') AND 
                               (SELECT d_date_sk FROM date_dim WHERE d_date = '2023-10-31')
    GROUP BY 
        ws_bill_cdemo_sk
),
CustomerDemographics AS (
    SELECT 
        cd_demo_sk,
        cd_gender,
        cd_marital_status,
        cd_education_status
    FROM 
        customer_demographics
),
HighValueCustomers AS (
    SELECT 
        s.CustomerDemoSK,
        c.cd_gender,
        c.cd_marital_status,
        c.cd_education_status
    FROM 
        SalesSummary s
    JOIN 
        CustomerDemographics c ON s.CustomerDemoSK = c.cd_demo_sk
    WHERE 
        s.TotalSales > (SELECT AVG(TotalSales) FROM SalesSummary)
)
SELECT 
    hvc.cd_gender,
    hvc.cd_marital_status,
    hvc.cd_education_status,
    COUNT(*) AS HighValueCustomerCount,
    SUM(ss.TotalSales) AS TotalHighValueSales
FROM 
    HighValueCustomers hvc
JOIN 
    SalesSummary ss ON hvc.CustomerDemoSK = ss.CustomerDemoSK
GROUP BY 
    hvc.cd_gender, hvc.cd_marital_status, hvc.cd_education_status
ORDER BY 
    TotalHighValueSales DESC;
