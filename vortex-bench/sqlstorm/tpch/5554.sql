
WITH RECURSIVE supplier_sales AS (
    SELECT s.s_suppkey, s.s_name, SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_sales
    FROM supplier s
    JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN lineitem l ON ps.ps_partkey = l.l_partkey
    GROUP BY s.s_suppkey, s.s_name
),
top_suppliers AS (
    SELECT s_suppkey, s_name, total_sales, ROW_NUMBER() OVER (ORDER BY total_sales DESC) AS sales_rank
    FROM supplier_sales
)
SELECT 
    s.s_name,
    c.c_name AS customer_name,
    o.o_orderkey,
    o.o_orderdate,
    l.l_quantity,
    l.l_extendedprice
FROM top_suppliers t
JOIN supplier s ON t.s_suppkey = s.s_suppkey
JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
JOIN lineitem l ON ps.ps_partkey = l.l_partkey
JOIN orders o ON l.l_orderkey = o.o_orderkey
JOIN customer c ON o.o_custkey = c.c_custkey
WHERE t.sales_rank <= 10
  AND o.o_orderdate BETWEEN DATE '1997-01-01' AND DATE '1997-12-31'
  AND l.l_returnflag = 'N'
ORDER BY t.total_sales DESC, o.o_orderdate ASC;
