SELECT
    l_orderkey,
    SUM(l_extendedprice * (1 - l_discount)) AS total_revenue,
    o_orderstatus,
    o_orderdate,
    c_nationkey,
    s_nationkey
FROM
    lineitem
JOIN
    orders ON lineitem.l_orderkey = orders.o_orderkey
JOIN
    customer ON orders.o_custkey = customer.c_custkey
JOIN
    partsupp ON lineitem.l_partkey = partsupp.ps_partkey
JOIN
    supplier ON partsupp.ps_suppkey = supplier.s_suppkey
GROUP BY
    l_orderkey, o_orderstatus, o_orderdate, c_nationkey, s_nationkey
ORDER BY
    total_revenue DESC
LIMIT 100;
