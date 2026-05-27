WITH RankedCustomers AS (
    SELECT 
        c.c_custkey,
        c.c_name,
        c.c_acctbal,
        n.n_name AS nation_name,
        DENSE_RANK() OVER (PARTITION BY n.n_name ORDER BY c.c_acctbal DESC) AS rank_per_nation
    FROM customer c
    JOIN nation n ON c.c_nationkey = n.n_nationkey
    WHERE c.c_acctbal > (SELECT AVG(c2.c_acctbal) FROM customer c2)
),
HighValueOrders AS (
    SELECT 
        o.o_orderkey,
        o.o_totalprice,
        o.o_orderdate,
        c.c_name AS customer_name,
        n.n_name AS nation_name
    FROM orders o
    JOIN customer c ON o.o_custkey = c.c_custkey
    JOIN nation n ON c.c_nationkey = n.n_nationkey
    WHERE o.o_totalprice > (SELECT AVG(o2.o_totalprice) FROM orders o2)
),
SupplierPartInfo AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        p.p_name,
        ps.ps_supplycost,
        ROW_NUMBER() OVER (PARTITION BY s.s_suppkey ORDER BY ps.ps_supplycost DESC) AS supply_rank
    FROM supplier s
    JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN part p ON ps.ps_partkey = p.p_partkey
)
SELECT 
    r.nation_name,
    SUM(HVO.o_totalprice) AS total_high_value_orders,
    COUNT(DISTINCT R.c_custkey) AS distinct_high_value_customers,
    COUNT(DISTINCT S.s_suppkey) AS distinct_suppliers,
    MAX(PA.ps_supplycost) AS max_supply_cost
FROM RankedCustomers R
LEFT JOIN HighValueOrders HVO ON HVO.nation_name = R.nation_name
LEFT JOIN SupplierPartInfo S ON S.s_suppkey = R.c_custkey
LEFT JOIN partsupp PA ON PA.ps_partkey = R.c_custkey
WHERE R.rank_per_nation <= 5
GROUP BY r.nation_name
ORDER BY total_high_value_orders DESC;
