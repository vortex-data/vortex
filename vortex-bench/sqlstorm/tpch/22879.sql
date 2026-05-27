WITH ranked_orders AS (
    SELECT o.o_orderkey,
           o.o_orderstatus,
           o.o_totalprice,
           o.o_orderdate,
           o.o_orderpriority,
           RANK() OVER (PARTITION BY o.o_orderstatus ORDER BY o.o_totalprice DESC) as order_rank
    FROM orders o
), 
filtered_parts AS (
    SELECT p.p_partkey,
           p.p_name,
           p.p_brand,
           p.p_retailprice,
           CASE 
               WHEN p.p_container IS NULL THEN 'UNKNOWN'
               ELSE p.p_container
           END AS container_type
    FROM part p
    WHERE p.p_size IN (SELECT DISTINCT CASE 
                                          WHEN l.l_quantity > 50 THEN 50 
                                          ELSE l.l_quantity 
                                      END 
                       FROM lineitem l WHERE l.l_returnflag = 'R')
), 
supplier_summary AS (
    SELECT s.s_suppkey,
           s.s_name,
           SUM(ps.ps_supplycost) AS total_supply_cost,
           COUNT(DISTINCT CASE WHEN ps.ps_availqty < 10 THEN ps.ps_partkey END) AS low_availability_parts
    FROM supplier s
    JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY s.s_suppkey, s.s_name
), 
combined_details AS (
    SELECT r.r_name AS region_name,
           n.n_name AS nation_name,
           COALESCE(ss.total_supply_cost, 0) AS total_supply_cost,
           COUNT(DISTINCT o.o_orderkey) FILTER (WHERE o.o_orderstatus = 'O') AS orders_open,
           COUNT(DISTINCT l.l_orderkey) AS total_lineitems
    FROM region r
    LEFT JOIN nation n ON r.r_regionkey = n.n_regionkey
    LEFT JOIN supplier_summary ss ON n.n_nationkey = ss.s_suppkey
    JOIN ranked_orders o ON o.o_orderkey IN (SELECT l.l_orderkey 
                                                FROM lineitem l 
                                                WHERE l.l_shipdate > cast('1998-10-01' as date) - INTERVAL '1 year')
    LEFT JOIN lineitem l ON o.o_orderkey = l.l_orderkey
    GROUP BY r.r_name, n.n_name, ss.total_supply_cost
)
SELECT cd.region_name,
       cd.nation_name,
       cd.total_supply_cost,
       cd.orders_open,
       cd.total_lineitems,
       STRING_AGG(DISTINCT CONCAT_WS(' - ', fp.p_name, fp.container_type), '; ') AS parts_info
FROM combined_details cd
LEFT JOIN filtered_parts fp ON fp.p_partkey IN (SELECT ps.ps_partkey 
                                                 FROM partsupp ps 
                                                 WHERE ps.ps_supplycost < 100.00)
WHERE cd.total_supply_cost IS NOT NULL
GROUP BY cd.region_name, cd.nation_name, cd.total_supply_cost, cd.orders_open, cd.total_lineitems
HAVING SUM(CASE WHEN cd.orders_open > 0 THEN 1 ELSE 0 END) > 1
ORDER BY cd.total_supply_cost DESC NULLS LAST;