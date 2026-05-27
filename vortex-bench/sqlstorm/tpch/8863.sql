WITH SupplierPartCost AS (
    SELECT s.s_suppkey, s.s_name, ps.ps_partkey, ps.ps_supplycost
    FROM supplier s
    JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
),
TotalCosts AS (
    SELECT spc.s_suppkey, spc.s_name, SUM(spc.ps_supplycost * ps.ps_availqty) AS total_cost
    FROM SupplierPartCost spc
    JOIN partsupp ps ON spc.ps_partkey = ps.ps_partkey
    GROUP BY spc.s_suppkey, spc.s_name
),
CustomerOrderTotals AS (
    SELECT c.c_custkey, c.c_name, SUM(o.o_totalprice) AS total_order_value
    FROM customer c
    JOIN orders o ON c.c_custkey = o.o_custkey
    GROUP BY c.c_custkey, c.c_name
)
SELECT cot.c_name AS customer_name, cot.total_order_value AS total_orders,
       tc.s_name AS supplier_name, tc.total_cost AS supplier_cost
FROM CustomerOrderTotals cot
JOIN TotalCosts tc ON cot.total_order_value > tc.total_cost
WHERE cot.total_order_value > 10000
ORDER BY cot.total_order_value DESC, tc.total_cost ASC;
