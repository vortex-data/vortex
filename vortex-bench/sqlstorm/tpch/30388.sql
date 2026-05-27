WITH RECURSIVE SupplierHierarchy AS (
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, 0 AS level
    FROM supplier s
    WHERE s.s_acctbal > (SELECT AVG(s_acctbal) FROM supplier)
    UNION ALL
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, sh.level + 1
    FROM supplier s
    JOIN SupplierHierarchy sh ON s.s_nationkey = sh.s_nationkey
    WHERE sh.level < 3
),
CustomerOrders AS (
    SELECT c.c_custkey, c.c_name, COUNT(o.o_orderkey) as total_orders
    FROM customer c
    LEFT JOIN orders o ON c.c_custkey = o.o_custkey
    WHERE c.c_acctbal IS NOT NULL
    GROUP BY c.c_custkey, c.c_name
),
PartSupplier AS (
    SELECT p.p_partkey, p.p_name, SUM(ps.ps_supplycost) AS total_supplycost
    FROM part p
    JOIN partsupp ps ON p.p_partkey = ps.ps_partkey
    GROUP BY p.p_partkey, p.p_name
),
TopNSuppliers AS (
    SELECT s.s_suppkey, s.s_name, SUM(ps.ps_availqty) AS total_availqty
    FROM supplier s
    JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY s.s_suppkey, s.s_name
    ORDER BY total_availqty DESC
    LIMIT 5
)
SELECT DISTINCT
    so.s_name AS supplier_name,
    co.c_name AS customer_name,
    po.p_name AS part_name,
    li.l_quantity,
    li.l_extendedprice,
    LI.l_discount,
    RANK() OVER (PARTITION BY co.c_custkey ORDER BY li.l_extendedprice DESC) AS price_rank,
    CASE
        WHEN li.l_returnflag = 'R' THEN 'Returned'
        ELSE 'Not Returned'
    END AS return_status
FROM lineitem li
JOIN orders o ON li.l_orderkey = o.o_orderkey
JOIN CustomerOrders co ON o.o_custkey = co.c_custkey
JOIN TopNSuppliers so ON li.l_suppkey = so.s_suppkey
JOIN PartSupplier po ON li.l_partkey = po.p_partkey
WHERE li.l_shipdate >= DATE '1997-01-01'
AND (li.l_discount > 0.1 OR co.total_orders > 5)
ORDER BY return_status, price_rank;