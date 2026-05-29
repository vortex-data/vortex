
WITH RankedSuppliers AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        s.s_acctbal,
        ROW_NUMBER() OVER (PARTITION BY s.s_nationkey ORDER BY s.s_acctbal DESC) AS rank
    FROM supplier s
),
PopularProducts AS (
    SELECT 
        p.p_partkey,
        p.p_name,
        SUM(l.l_quantity) AS total_quantity
    FROM part p
    JOIN lineitem l ON p.p_partkey = l.l_partkey
    GROUP BY p.p_partkey, p.p_name
    HAVING SUM(l.l_quantity) > 100
),
CustomerOrders AS (
    SELECT 
        c.c_custkey, 
        c.c_name,
        COUNT(o.o_orderkey) AS order_count,
        SUM(o.o_totalprice) AS total_spent
    FROM customer c
    LEFT JOIN orders o ON c.c_custkey = o.o_custkey
    GROUP BY c.c_custkey, c.c_name
    HAVING SUM(o.o_totalprice) > 1000
)
SELECT 
    cu.c_name,
    cu.order_count,
    cu.total_spent,
    rs.s_name AS top_supplier,
    pp.p_name AS popular_product,
    pp.total_quantity,
    COALESCE(rs.s_acctbal, 0) AS supplier_balance
FROM CustomerOrders cu
LEFT JOIN RankedSuppliers rs ON cu.order_count = rs.rank 
LEFT JOIN PopularProducts pp ON rs.s_suppkey = pp.p_partkey
INNER JOIN supplier s ON s.s_nationkey = cu.c_custkey
WHERE cu.order_count > 5 AND pp.total_quantity IS NOT NULL
ORDER BY cu.total_spent DESC, cu.c_name;
