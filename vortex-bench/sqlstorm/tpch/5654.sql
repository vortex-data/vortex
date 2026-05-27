
WITH SupplierCost AS (
    SELECT s.s_suppkey, SUM(ps.ps_supplycost * ps.ps_availqty) AS total_cost
    FROM supplier s
    JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY s.s_suppkey
),
NationSummary AS (
    SELECT n.n_nationkey, n.n_name, SUM(o.o_totalprice) AS total_sales
    FROM nation n
    JOIN supplier s ON n.n_nationkey = s.s_nationkey
    JOIN customer c ON s.s_suppkey = c.c_custkey
    JOIN orders o ON c.c_custkey = o.o_custkey
    GROUP BY n.n_nationkey, n.n_name
),
PartDetail AS (
    SELECT p.p_partkey, p.p_name, COUNT(l.l_linenumber) AS total_lines, AVG(l.l_extendedprice) AS avg_price
    FROM part p
    JOIN lineitem l ON p.p_partkey = l.l_partkey
    GROUP BY p.p_partkey, p.p_name
)
SELECT ns.n_name, pd.p_name, SUM(ns.total_sales) AS total_sales, SUM(sc.total_cost) AS total_cost, MAX(pd.avg_price) AS max_avg_price
FROM NationSummary ns
JOIN SupplierCost sc ON ns.n_nationkey = sc.s_suppkey
JOIN PartDetail pd ON pd.p_partkey = sc.s_suppkey
GROUP BY ns.n_name, pd.p_name
ORDER BY total_sales DESC, total_cost DESC
LIMIT 10;
