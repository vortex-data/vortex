WITH RECURSIVE CTE_Suppliers AS (
    SELECT s_suppkey, s_name, n_name, s_acctbal, 
           ROW_NUMBER() OVER (PARTITION BY n_name ORDER BY s_acctbal DESC) AS rank
    FROM supplier
    JOIN nation ON supplier.s_nationkey = nation.n_nationkey
),
CTE_OrderSummary AS (
    SELECT o_custkey, SUM(o_totalprice) AS total_spent
    FROM orders
    GROUP BY o_custkey
),
CTE_PartPricing AS (
    SELECT ps_partkey, AVG(ps_supplycost) AS avg_supplycost
    FROM partsupp
    GROUP BY ps_partkey
)
SELECT p.p_name, 
       COALESCE(s.s_name, 'No Supplier') AS supplier_name, 
       ns.total_spent, 
       pp.avg_supplycost,
       p.p_retailprice - COALESCE(pp.avg_supplycost, 0) AS profit_margin,
       CASE 
           WHEN pp.avg_supplycost IS NULL THEN 'No Cost Data'
           WHEN (p.p_retailprice - COALESCE(pp.avg_supplycost, 0)) < 0 THEN 'Loss'
           ELSE 'Profit'
       END AS profitability_status
FROM part p
LEFT JOIN CTE_PartPricing pp ON p.p_partkey = pp.ps_partkey
LEFT JOIN CTE_Suppliers s ON s.rank = 1 AND p.p_partkey = s.s_suppkey
LEFT JOIN CTE_OrderSummary ns ON ns.o_custkey = p.p_partkey
WHERE (p.p_size >= 10 OR p.p_comment IS NULL)
  AND (pp.avg_supplycost < p.p_retailprice OR pp.avg_supplycost IS NULL)
ORDER BY profit_margin DESC
LIMIT 100;
