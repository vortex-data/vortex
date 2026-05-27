WITH Supplier_Count AS (
    SELECT p.p_partkey, COUNT(DISTINCT s.s_suppkey) AS supplier_count
    FROM part p
    JOIN partsupp ps ON p.p_partkey = ps.ps_partkey
    JOIN supplier s ON ps.ps_suppkey = s.s_suppkey
    GROUP BY p.p_partkey
),
Customer_Count AS (
    SELECT o.o_orderkey, COUNT(DISTINCT c.c_custkey) AS customer_count
    FROM orders o
    JOIN customer c ON o.o_custkey = c.c_custkey
    GROUP BY o.o_orderkey
),
LineItem_Stats AS (
    SELECT l.l_orderkey, 
           SUM(l.l_extendedprice) AS total_revenue, 
           AVG(l.l_discount) AS avg_discount, 
           MAX(l.l_tax) AS max_tax
    FROM lineitem l
    GROUP BY l.l_orderkey
)
SELECT p.p_name, 
       sc.supplier_count, 
       cc.customer_count, 
       ls.total_revenue, 
       ls.avg_discount, 
       ls.max_tax
FROM Supplier_Count sc
JOIN Customer_Count cc ON sc.p_partkey = cc.o_orderkey
JOIN LineItem_Stats ls ON cc.o_orderkey = ls.l_orderkey
JOIN part p ON sc.p_partkey = p.p_partkey
WHERE sc.supplier_count > 5 
AND cc.customer_count > 10 
AND ls.total_revenue > 10000
ORDER BY ls.total_revenue DESC;
