WITH RankedOrders AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        o.o_totalprice,
        c.c_mktsegment,
        RANK() OVER (PARTITION BY c.c_mktsegment ORDER BY o.o_totalprice DESC) AS segment_rank
    FROM 
        orders o
    JOIN 
        customer c ON o.o_custkey = c.c_custkey
    WHERE 
        o.o_orderdate >= DATE '1997-01-01' 
        AND o.o_orderdate < DATE '1998-01-01'
),
SupplierPartSummary AS (
    SELECT 
        ps.ps_partkey,
        s.s_nationkey,
        SUM(ps.ps_availqty) AS total_avail_qty,
        SUM(ps.ps_supplycost * ps.ps_availqty) AS total_cost
    FROM 
        partsupp ps
    JOIN 
        supplier s ON ps.ps_suppkey = s.s_suppkey
    GROUP BY 
        ps.ps_partkey, s.s_nationkey
    HAVING 
        SUM(ps.ps_availqty) > 100
),
HighValueOrders AS (
    SELECT 
        ro.o_orderkey,
        ro.o_totalprice,
        ro.o_orderdate,
        ro.c_mktsegment
    FROM 
        RankedOrders ro
    WHERE 
        ro.segment_rank <= 5
)
SELECT 
    hvo.o_orderkey,
    hvo.o_totalprice,
    hvo.o_orderdate,
    n.n_name AS nation,
    pp.p_name AS part_name,
    COALESCE(sp.total_avail_qty, 0) AS available_quantity,
    COALESCE(sp.total_cost, 0) AS total_cost
FROM 
    HighValueOrders hvo
LEFT JOIN 
    lineitem l ON hvo.o_orderkey = l.l_orderkey
LEFT JOIN 
    partsupp ps ON l.l_partkey = ps.ps_partkey
LEFT JOIN 
    supplier s ON ps.ps_suppkey = s.s_suppkey
LEFT JOIN 
    nation n ON s.s_nationkey = n.n_nationkey
LEFT JOIN 
    part pp ON ps.ps_partkey = pp.p_partkey
LEFT JOIN 
    SupplierPartSummary sp ON ps.ps_partkey = sp.ps_partkey AND s.s_nationkey = sp.s_nationkey
WHERE 
    hvo.o_totalprice > (SELECT AVG(o_totalprice) FROM orders)
ORDER BY 
    hvo.o_orderdate DESC, hvo.o_totalprice DESC;