
WITH DailySales AS (
    SELECT 
        dd.d_date AS SaleDate,
        SUM(ws.ws_sales_price * ws.ws_quantity) AS TotalSales,
        COUNT(DISTINCT ws.ws_order_number) AS TotalOrders,
        AVG(ws.ws_sales_price) AS AvgOrderValue
    FROM 
        web_sales ws
    JOIN 
        date_dim dd ON ws.ws_sold_date_sk = dd.d_date_sk
    GROUP BY 
        dd.d_date
),
TopCustomers AS (
    SELECT 
        c.c_customer_id,
        SUM(ws.ws_sales_price * ws.ws_quantity) AS CustomerTotalSpent
    FROM 
        customer c
    JOIN 
        web_sales ws ON c.c_customer_sk = ws.ws_bill_customer_sk
    GROUP BY 
        c.c_customer_id
    ORDER BY 
        CustomerTotalSpent DESC
    LIMIT 10
),
SalesWithRanking AS (
    SELECT 
        ds.SaleDate,
        ds.TotalSales,
        ds.TotalOrders,
        ds.AvgOrderValue,
        RANK() OVER (ORDER BY ds.TotalSales DESC) AS SalesRank
    FROM 
        DailySales ds
)
SELECT 
    s.SaleDate,
    s.TotalSales,
    s.TotalOrders,
    s.AvgOrderValue,
    tc.c_customer_id AS TopCustomer,
    tc.CustomerTotalSpent,
    CASE 
        WHEN s.AvgOrderValue IS NULL THEN 'No Sales'
        WHEN s.AvgOrderValue < 100 THEN 'Low'
        ELSE 'High'
    END AS SalesCategory
FROM 
    SalesWithRanking s
LEFT JOIN 
    TopCustomers tc ON tc.CustomerTotalSpent BETWEEN 500 AND 10000
WHERE 
    s.SalesRank <= 5 OR tc.c_customer_id IS NOT NULL
ORDER BY 
    s.TotalSales DESC, TopCustomer DESC;
