WITH RECURSIVE SupplierHierarchy AS (
    SELECT s.s_suppkey, s.s_name, s.s_acctbal, 0 AS level
    FROM supplier s
    WHERE s.s_acctbal > 100000
    UNION ALL
    SELECT s.s_suppkey, s.s_name, s.s_acctbal, sh.level + 1
    FROM supplier s
    INNER JOIN SupplierHierarchy sh ON s.s_suppkey = sh.s_suppkey
    WHERE s.s_acctbal > 150000
),
OrderStats AS (
    SELECT o.o_custkey, SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_sales,
           COUNT(DISTINCT o.o_orderkey) AS order_count
    FROM orders o
    JOIN lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE o.o_orderstatus = 'O'
    GROUP BY o.o_custkey
),
CustomerSales AS (
    SELECT c.c_custkey, c.c_name, cs.total_sales, cs.order_count,
           RANK() OVER (ORDER BY cs.total_sales DESC) AS sales_rank
    FROM customer c
    LEFT JOIN OrderStats cs ON c.c_custkey = cs.o_custkey
    WHERE c.c_acctbal IS NOT NULL
)
SELECT n.n_name, 
       COALESCE(SUM(CASE WHEN cs.sales_rank <= 10 THEN cs.total_sales ELSE 0 END), 0) AS top_sales,
       COUNT(DISTINCT cs.c_custkey) AS customer_count
FROM nation n
LEFT JOIN customer c ON n.n_nationkey = c.c_nationkey
LEFT JOIN CustomerSales cs ON c.c_custkey = cs.c_custkey
WHERE n.n_name IS NOT NULL
GROUP BY n.n_name
HAVING COUNT(DISTINCT cs.c_custkey) > 5
ORDER BY top_sales DESC
LIMIT 5;

