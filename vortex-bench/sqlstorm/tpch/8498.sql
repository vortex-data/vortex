WITH RankedOrders AS (
    SELECT
        o.o_orderkey,
        o.o_orderdate,
        o.o_totalprice,
        c.c_mktsegment,
        ROW_NUMBER() OVER (PARTITION BY c.c_mktsegment ORDER BY o.o_totalprice DESC) AS rn
    FROM
        orders o
    JOIN
        customer c ON o.o_custkey = c.c_custkey
),
TopOrderSegments AS (
    SELECT
        ro.c_mktsegment,
        ro.o_orderkey,
        ro.o_orderdate,
        ro.o_totalprice
    FROM
        RankedOrders ro
    WHERE
        ro.rn <= 10
)
SELECT
    p.p_name,
    SUM(l.l_quantity) AS total_quantity,
    SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue,
    COUNT(DISTINCT o.o_orderkey) AS order_count,
    n.n_name AS supplier_nation
FROM
    lineitem l
JOIN
    orders o ON l.l_orderkey = o.o_orderkey
JOIN
    partsupp ps ON l.l_partkey = ps.ps_partkey
JOIN
    supplier s ON ps.ps_suppkey = s.s_suppkey
JOIN
    nation n ON s.s_nationkey = n.n_nationkey
JOIN
    TopOrderSegments tos ON o.o_orderkey = tos.o_orderkey
JOIN
    part p ON l.l_partkey = p.p_partkey
WHERE
    l.l_shipdate >= DATE '1996-01-01'
    AND l.l_shipdate < DATE '1997-01-01'
GROUP BY
    p.p_name,
    n.n_name
HAVING
    SUM(l.l_extendedprice * (1 - l.l_discount)) > 10000
ORDER BY
    revenue DESC, total_quantity DESC;