WITH SupplierParts AS (
    SELECT s.s_suppkey, s.s_name, p.p_partkey, p.p_name, ps.ps_supplycost, ps.ps_availqty
    FROM supplier s
    JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN part p ON ps.ps_partkey = p.p_partkey
    WHERE ps.ps_availqty > 0
),
OrderDetails AS (
    SELECT o.o_orderkey, o.o_orderdate, c.c_custkey, SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue
    FROM orders o
    JOIN customer c ON o.o_custkey = c.c_custkey
    JOIN lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE o.o_orderdate >= '1997-01-01' AND o.o_orderdate < '1998-01-01'
    GROUP BY o.o_orderkey, o.o_orderdate, c.c_custkey
),
RankedSuppliers AS (
    SELECT s.s_suppkey, s.s_name, COUNT(DISTINCT op.o_orderkey) AS order_count
    FROM SupplierParts s
    JOIN OrderDetails op ON s.s_suppkey = op.o_orderkey
    GROUP BY s.s_suppkey, s.s_name
    HAVING COUNT(DISTINCT op.o_orderkey) > 5
),
TopSuppliers AS (
    SELECT *, RANK() OVER (ORDER BY order_count DESC) AS supplier_rank
    FROM RankedSuppliers
)
SELECT ts.s_suppkey, ts.s_name, ts.order_count, sp.p_partkey, sp.p_name, sp.ps_supplycost, sp.ps_availqty
FROM TopSuppliers ts
JOIN SupplierParts sp ON ts.s_suppkey = sp.s_suppkey
WHERE ts.supplier_rank <= 10
ORDER BY ts.order_count DESC, sp.ps_supplycost ASC;