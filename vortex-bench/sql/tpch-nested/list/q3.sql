select
    l_orderkey,
    sum(lineitem_data.l_extendedprice * (1 - lineitem_data.l_discount)) as revenue,
    o_orderdate,
    orders_data.o_shippriority
from
    customer,
    (select o_orderkey, o_custkey, o_orderdate, unnest(orders_data) as orders_data from orders) as orders,
    (select l_orderkey, l_partkey, l_suppkey, l_shipdate, unnest(lineitem_data) as lineitem_data from lineitem) as lineitem
where
    c_mktsegment = 'BUILDING'
    and c_custkey = o_custkey
    and l_orderkey = o_orderkey
    and o_orderdate < date '1995-03-15'
    and l_shipdate > date '1995-03-15'
group by
    l_orderkey,
    o_orderdate,
    orders_data.o_shippriority
order by
    revenue desc,
    o_orderdate
limit 10;
