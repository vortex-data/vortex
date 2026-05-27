WITH RECURSIVE region_supplier AS (
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, s.s_acctbal, s.s_comment
    FROM supplier s
    INNER JOIN nation n ON s.s_nationkey = n.n_nationkey
    WHERE n.n_name = 'Canada'
    
    UNION ALL
    
    SELECT s.s_suppkey, s.s_name, s.s_nationkey, s.s_acctbal, s.s_comment
    FROM supplier s
    INNER JOIN region_supplier rs ON rs.s_nationkey = s.s_nationkey
    WHERE rs.s_acctbal > 5000
),
order_summary AS (
    SELECT o.o_orderkey, SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue
    FROM orders o
    JOIN lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE l.l_shipdate >= DATE '1997-01-01' AND l.l_shipdate <= DATE '1997-12-31'
    GROUP BY o.o_orderkey
),
market_analysis AS (
    SELECT c.c_mktsegment, COUNT(DISTINCT o.o_orderkey) AS order_count,
           SUM(os.total_revenue) AS segment_revenue
    FROM customer c
    LEFT JOIN orders o ON c.c_custkey = o.o_custkey
    LEFT JOIN order_summary os ON o.o_orderkey = os.o_orderkey
    GROUP BY c.c_mktsegment
)
SELECT r.r_name, COALESCE(SUM(ma.segment_revenue), 0) AS total_segment_revenue,
       COUNT(DISTINCT rs.s_suppkey) AS supplier_count, 
       AVG(rs.s_acctbal) AS average_acctbal
FROM region r
LEFT JOIN nation n ON r.r_regionkey = n.n_regionkey
LEFT JOIN region_supplier rs ON n.n_nationkey = rs.s_nationkey
LEFT JOIN market_analysis ma ON ma.c_mktsegment = 'BUILDING'
GROUP BY r.r_name
HAVING AVG(rs.s_acctbal) > (SELECT AVG(s.s_acctbal) FROM supplier s WHERE s.s_acctbal IS NOT NULL)
ORDER BY total_segment_revenue DESC;