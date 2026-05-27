WITH RECURSIVE SupplierHierarchy AS (
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, 1 AS level
    FROM supplier s
    WHERE s.s_acctbal > 50000
    UNION ALL
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, sh.level + 1
    FROM supplier s
    JOIN SupplierHierarchy sh ON s.s_suppkey = sh.s_nationkey
)
SELECT 
    SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue,
    n.n_name AS nation_name,
    COUNT(DISTINCT o.o_orderkey) AS total_orders,
    COUNT(DISTINCT c.c_custkey) AS total_customers,
    MIN(o.o_orderdate) AS first_order_date,
    MAX(o.o_orderdate) AS last_order_date,
    CASE 
        WHEN COUNT(DISTINCT o.o_orderkey) > 1000 THEN 'High'
        WHEN COUNT(DISTINCT o.o_orderkey) BETWEEN 500 AND 1000 THEN 'Medium'
        ELSE 'Low' 
    END AS order_volume_category
FROM 
    lineitem l
JOIN 
    orders o ON l.l_orderkey = o.o_orderkey
JOIN 
    customer c ON o.o_custkey = c.c_custkey
JOIN 
    nation n ON c.c_nationkey = n.n_nationkey
JOIN 
    partsupp ps ON l.l_partkey = ps.ps_partkey
JOIN 
    SupplierHierarchy sh ON ps.ps_suppkey = sh.s_suppkey
WHERE 
    l.l_shipdate BETWEEN '1997-01-01' AND '1997-12-31'
    AND n.n_regionkey IN (SELECT r.r_regionkey FROM region r WHERE r.r_name LIKE '%NA%')
    AND l.l_returnflag = 'N'
GROUP BY 
    n.n_name
ORDER BY 
    total_revenue DESC
LIMIT 10;