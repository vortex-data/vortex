WITH RankedSuppliers AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        s.s_acctbal,
        ROW_NUMBER() OVER (PARTITION BY s.s_nationkey ORDER BY s.s_acctbal DESC) AS rnk,
        SUM(ps.ps_supplycost) OVER (PARTITION BY s.s_nationkey) AS total_supply_cost
    FROM 
        supplier s
    JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
),
SubQueryCustomer AS (
    SELECT 
        c.c_custkey,
        COUNT(o.o_orderkey) AS order_count
    FROM 
        customer c
    LEFT JOIN orders o ON c.c_custkey = o.o_custkey
    GROUP BY c.c_custkey
    HAVING COUNT(o.o_orderkey) > (SELECT AVG(order_count) FROM (SELECT COUNT(o.o_orderkey) AS order_count FROM orders o GROUP BY o.o_custkey) AS sub_avg)
),
FilteredParts AS (
    SELECT 
        p.p_partkey,
        p.p_name,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue
    FROM 
        part p
    JOIN lineitem l ON p.p_partkey = l.l_partkey
    GROUP BY p.p_partkey, p.p_name
    HAVING SUM(l.l_extendedprice * (1 - l.l_discount)) > 50000
)
SELECT 
    r.r_name,
    COUNT(DISTINCT cs.c_custkey) AS customer_count,
    SUM(fp.revenue) AS total_revenue,
    MAX(rs.total_supply_cost) AS max_supply_cost
FROM 
    region r
LEFT JOIN nation n ON r.r_regionkey = n.n_regionkey
LEFT JOIN RankedSuppliers rs ON n.n_nationkey = rs.s_suppkey 
LEFT JOIN SubQueryCustomer cs ON cs.c_custkey = rs.s_suppkey
LEFT JOIN FilteredParts fp ON fp.p_partkey = rs.s_suppkey
WHERE 
    r.r_name IS NOT NULL 
    AND (rs.rnk = 1 OR rs.total_supply_cost IS NULL)
GROUP BY 
    r.r_name
HAVING 
    SUM(fp.revenue) IS NOT NULL
    AND COUNT(DISTINCT cs.c_custkey) > 0
ORDER BY 
    customer_count DESC, total_revenue ASC;
