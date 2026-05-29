WITH RankedOrders AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue,
        RANK() OVER (PARTITION BY EXTRACT(YEAR FROM o.o_orderdate) ORDER BY SUM(l.l_extendedprice * (1 - l.l_discount)) DESC) AS revenue_rank
    FROM 
        orders o
    JOIN 
        lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE 
        o.o_orderdate >= DATE '1996-01-01' AND o.o_orderdate < DATE '1997-01-01'
    GROUP BY 
        o.o_orderkey, o.o_orderdate
),
TopCustomers AS (
    SELECT 
        c.c_custkey,
        c.c_name,
        SUM(lo.total_revenue) AS customer_revenue
    FROM 
        customer c
    JOIN 
        RankedOrders lo ON c.c_custkey = lo.o_orderkey
    GROUP BY 
        c.c_custkey, c.c_name
    HAVING 
        SUM(lo.total_revenue) > 100000
)
SELECT 
    rc.r_name AS region,
    nc.n_name AS nation,
    COUNT(tc.c_custkey) AS top_customer_count,
    SUM(tc.customer_revenue) AS total_customer_revenue
FROM 
    region rc
JOIN 
    nation nc ON rc.r_regionkey = nc.n_regionkey
JOIN 
    supplier s ON s.s_nationkey = nc.n_nationkey
JOIN 
    partsupp ps ON ps.ps_suppkey = s.s_suppkey
JOIN 
    RankedOrders lo ON lo.o_orderkey = ps.ps_partkey
JOIN 
    TopCustomers tc ON tc.c_custkey = lo.o_orderkey
GROUP BY 
    rc.r_name, nc.n_name
ORDER BY 
    total_customer_revenue DESC;