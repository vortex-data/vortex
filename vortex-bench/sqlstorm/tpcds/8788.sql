WITH CustomerSales AS (
    SELECT 
        c.c_customer_sk,
        c.c_first_name,
        c.c_last_name,
        SUM(ws.ws_ext_sales_price) AS total_sales,
        COUNT(ws.ws_order_number) AS order_count
    FROM 
        customer c
    JOIN 
        web_sales ws ON c.c_customer_sk = ws.ws_bill_customer_sk
    WHERE 
        c.c_birth_year >= 1980 
    GROUP BY 
        c.c_customer_sk, c.c_first_name, c.c_last_name
),
TopCustomers AS (
    SELECT 
        c.c_customer_sk,
        c.c_first_name,
        c.c_last_name,
        cs.total_sales,
        cs.order_count,
        RANK() OVER (ORDER BY cs.total_sales DESC) AS sales_rank
    FROM 
        CustomerSales cs
    JOIN 
        customer c ON cs.c_customer_sk = c.c_customer_sk
)
SELECT 
    tc.c_customer_sk,
    tc.c_first_name,
    tc.c_last_name,
    tc.total_sales,
    tc.order_count,
    d.d_date AS last_order_date,
    i.i_item_desc,
    s.s_store_name
FROM 
    TopCustomers tc
JOIN 
    store_sales ss ON ss.ss_customer_sk = tc.c_customer_sk
JOIN 
    date_dim d ON d.d_date_sk = ss.ss_sold_date_sk 
JOIN 
    item i ON i.i_item_sk = ss.ss_item_sk
JOIN 
    store s ON s.s_store_sk = ss.ss_store_sk
WHERE 
    tc.sales_rank <= 10 
    AND d.d_year = 2000 
ORDER BY 
    tc.total_sales DESC;