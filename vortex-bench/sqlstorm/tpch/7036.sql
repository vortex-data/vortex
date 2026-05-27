WITH RankedSuppliers AS (
    SELECT s.s_suppkey, s.s_name, s.s_acctbal,
           ROW_NUMBER() OVER (PARTITION BY n.n_nationkey ORDER BY s.s_acctbal DESC) AS rank
    FROM supplier s
    JOIN nation n ON s.s_nationkey = n.n_nationkey
    WHERE s.s_acctbal > (
        SELECT AVG(s2.s_acctbal) 
        FROM supplier s2 
        WHERE s2.s_nationkey = s.s_nationkey
    )
),
PartOrders AS (
    SELECT l.l_orderkey, p.p_partkey, SUM(l.l_quantity) AS total_quantity
    FROM lineitem l
    JOIN partsupp ps ON l.l_partkey = ps.ps_partkey
    JOIN part p ON ps.ps_partkey = p.p_partkey
    GROUP BY l.l_orderkey, p.p_partkey
),
OrderTotal AS (
    SELECT o.o_orderkey, SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_price
    FROM orders o
    JOIN lineitem l ON o.o_orderkey = l.l_orderkey
    GROUP BY o.o_orderkey
),
CustomerSpend AS (
    SELECT c.c_custkey, SUM(ot.total_price) AS spent
    FROM customer c
    JOIN OrderTotal ot ON c.c_custkey = ot.o_orderkey
    GROUP BY c.c_custkey
)
SELECT rs.s_name, rs.s_acctbal, ps.total_quantity, cs.spent
FROM RankedSuppliers rs
JOIN PartOrders ps ON rs.s_suppkey = ps.p_partkey
JOIN CustomerSpend cs ON cs.c_custkey = ps.l_orderkey
WHERE rs.rank <= 5
ORDER BY rs.s_acctbal DESC, cs.spent DESC;
