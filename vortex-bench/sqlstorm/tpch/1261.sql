WITH RankedOrders AS (
    SELECT o.o_orderkey,
           o.o_orderdate,
           o.o_totalprice,
           o.o_orderstatus,
           ROW_NUMBER() OVER (PARTITION BY o.o_orderstatus ORDER BY o.o_orderdate DESC) AS rn
    FROM orders o
),
SupplierSummary AS (
    SELECT ps.ps_partkey,
           s.s_nationkey,
           SUM(ps.ps_supplycost * ps.ps_availqty) AS total_supply_cost,
           AVG(s.s_acctbal) AS avg_supplier_balance
    FROM partsupp ps
    JOIN supplier s ON ps.ps_suppkey = s.s_suppkey
    GROUP BY ps.ps_partkey, s.s_nationkey
),
CustomerRegion AS (
    SELECT c.c_custkey,
           c.c_name,
           n.n_regionkey,
           r.r_name,
           ROW_NUMBER() OVER (PARTITION BY r.r_name ORDER BY c.c_acctbal DESC) AS region_rank
    FROM customer c
    JOIN nation n ON c.c_nationkey = n.n_nationkey
    JOIN region r ON n.n_regionkey = r.r_regionkey
),
LineItemDetails AS (
    SELECT l.l_orderkey,
           SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue,
           COUNT(*) AS item_count
    FROM lineitem l
    WHERE l.l_shipdate >= '1997-01-01' AND l.l_shipdate < '1998-01-01'
    GROUP BY l.l_orderkey
)
SELECT cr.r_name,
       COUNT(DISTINCT co.o_orderkey) AS total_orders,
       SUM(ss.total_supply_cost) AS total_supply_cost,
       AVG(ss.avg_supplier_balance) AS average_supplier_balance,
       SUM(ld.revenue) AS total_revenue,
       SUM(ld.item_count) AS total_items
FROM CustomerRegion cr
LEFT JOIN RankedOrders co ON cr.c_custkey = co.o_orderkey
LEFT JOIN SupplierSummary ss ON cr.n_regionkey = ss.s_nationkey
LEFT JOIN LineItemDetails ld ON co.o_orderkey = ld.l_orderkey
WHERE cr.region_rank <= 10
GROUP BY cr.r_name
ORDER BY total_revenue DESC;