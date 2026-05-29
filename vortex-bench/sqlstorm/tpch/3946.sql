WITH RankedOrders AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        o.o_totalprice,
        c.c_name,
        ROW_NUMBER() OVER (PARTITION BY c.c_nationkey ORDER BY o.o_totalprice DESC) AS order_rank
    FROM 
        orders o
    JOIN 
        customer c ON o.o_custkey = c.c_custkey
    WHERE 
        o.o_orderdate >= DATE '1997-01-01' AND o.o_orderdate < DATE '1998-01-01'
),
TopCustomerOrders AS (
    SELECT 
        r.r_name AS region,
        n.n_name AS nation,
        ro.c_name AS customer_name,
        ro.o_orderkey,
        ro.o_orderdate,
        ro.o_totalprice
    FROM 
        RankedOrders ro
    JOIN 
        customer c ON ro.c_name = c.c_name
    JOIN 
        nation n ON c.c_nationkey = n.n_nationkey
    JOIN 
        region r ON n.n_regionkey = r.r_regionkey
    WHERE 
        ro.order_rank <= 3
),
AvgOrderPrice AS (
    SELECT 
        nu.n_name,
        AVG(tco.o_totalprice) AS average_price
    FROM 
        TopCustomerOrders tco
    JOIN 
        nation nu ON tco.nation = nu.n_name
    GROUP BY 
        nu.n_name
)
SELECT 
    tc.region,
    tc.nation,
    tc.customer_name,
    tc.o_orderkey,
    tc.o_orderdate,
    tc.o_totalprice,
    aop.average_price,
    CASE 
        WHEN tc.o_totalprice > aop.average_price THEN 'Above Average'
        WHEN tc.o_totalprice < aop.average_price THEN 'Below Average'
        ELSE 'Average' 
    END AS price_comparison
FROM 
    TopCustomerOrders tc
LEFT JOIN 
    AvgOrderPrice aop ON tc.nation = aop.n_name
ORDER BY 
    tc.region, 
    tc.nation, 
    tc.o_totalprice DESC;