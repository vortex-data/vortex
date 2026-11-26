select
    orders_data.o_orderpriority,
    count(*) as order_count
from
    (select o_orderkey, o_custkey, o_orderdate, unnest(orders_data) as orders_data from orders) as orders
where
    o_orderdate >= date '1993-07-01'
    and o_orderdate < date '1993-07-01' + interval '3' month
    and exists (
        select
            *
        from
            (select l_orderkey, l_partkey, l_suppkey, l_shipdate, unnest(lineitem_data) as lineitem_data from lineitem) as lineitem
        where
            l_orderkey = o_orderkey
            and lineitem_data.l_commitdate < lineitem_data.l_receiptdate
    )
group by
    orders_data.o_orderpriority
order by
    orders_data.o_orderpriority;
