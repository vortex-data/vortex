WITH RECURSIVE nation_hierarchy AS (
    SELECT n_nationkey, n_name, n_regionkey, 0 AS level
    FROM nation
    WHERE n_regionkey IS NOT NULL
    UNION ALL
    SELECT n.n_nationkey, n.n_name, n.n_regionkey, nh.level + 1
    FROM nation n
    JOIN nation_hierarchy nh ON n.n_nationkey = nh.n_regionkey
),
supplier_costs AS (
    SELECT s.s_suppkey, SUM(ps.ps_supplycost * ps.ps_availqty) AS total_cost
    FROM supplier s
    JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY s.s_suppkey
),
order_summary AS (
    SELECT o.o_orderkey, o.o_custkey, SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_lineitem_price
    FROM orders o
    JOIN lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE l.l_shipdate >= '1997-01-01' AND l.l_shipdate < '1998-01-01'
    GROUP BY o.o_orderkey, o.o_custkey
),
customer_rank AS (
    SELECT c.c_custkey, CUME_DIST() OVER (ORDER BY SUM(os.total_lineitem_price) DESC) AS customer_rank
    FROM customer c
    JOIN order_summary os ON c.c_custkey = os.o_custkey
    GROUP BY c.c_custkey
)
SELECT r.r_name AS region_name, 
       n.n_name AS nation_name, 
       s.s_name AS supplier_name,
       ROUND(COALESCE(s_cost.total_cost, 0), 2) AS supplier_total_cost,
       ROUND(SUM(os.total_lineitem_price), 2) AS total_order_value,
       cr.customer_rank
FROM region r
LEFT JOIN nation n ON r.r_regionkey = n.n_regionkey
LEFT JOIN supplier s ON n.n_nationkey = s.s_nationkey
LEFT JOIN supplier_costs s_cost ON s.s_suppkey = s_cost.s_suppkey
LEFT JOIN order_summary os ON s.s_suppkey = os.o_custkey
LEFT JOIN customer_rank cr ON os.o_custkey = cr.c_custkey
GROUP BY r.r_name, n.n_name, s.s_name, s_cost.total_cost, cr.customer_rank
HAVING SUM(os.total_lineitem_price) IS NOT NULL
ORDER BY region_name, nation_name, supplier_total_cost DESC;