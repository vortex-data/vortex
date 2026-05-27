WITH RankedOrders AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue,
        RANK() OVER (PARTITION BY o.o_orderdate ORDER BY SUM(l.l_extendedprice * (1 - l.l_discount)) DESC) AS order_rank
    FROM 
        orders o
    JOIN 
        lineitem l ON o.o_orderkey = l.l_orderkey
    GROUP BY 
        o.o_orderkey, o.o_orderdate
),
TopOrders AS (
    SELECT 
        r.o_orderkey,
        r.o_orderdate,
        r.total_revenue
    FROM 
        RankedOrders r
    WHERE 
        r.order_rank <= 10
)
SELECT 
    o.o_orderkey,
    o.o_orderdate,
    o.total_revenue,
    c.c_name,
    s.s_name,
    p.p_name,
    ps.ps_availqty
FROM 
    TopOrders o
JOIN 
    orders ord ON o.o_orderkey = ord.o_orderkey
JOIN 
    lineitem l ON ord.o_orderkey = l.l_orderkey
JOIN 
    partsupp ps ON l.l_partkey = ps.ps_partkey AND l.l_suppkey = ps.ps_suppkey
JOIN 
    supplier s ON ps.ps_suppkey = s.s_suppkey
JOIN 
    customer c ON ord.o_custkey = c.c_custkey
JOIN 
    part p ON l.l_partkey = p.p_partkey
WHERE 
    c.c_acctbal > 10000 AND 
    c.c_mktsegment = 'BUILDING'
ORDER BY 
    o.total_revenue DESC, 
    o.o_orderdate ASC;
