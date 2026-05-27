
SELECT COUNT(*) AS total_orders, 
       SUM(o.o_totalprice) AS total_revenue, 
       AVG(l.l_extendedprice) AS avg_lineitem_price, 
       COUNT(DISTINCT c.c_custkey) AS total_customers
FROM orders o
JOIN lineitem l ON o.o_orderkey = l.l_orderkey
JOIN customer c ON o.o_custkey = c.c_custkey
WHERE o.o_orderstatus = 'O'
  AND l.l_shipdate BETWEEN DATE '1997-01-01' AND DATE '1997-12-31'
GROUP BY o.o_orderpriority
ORDER BY total_revenue DESC;
