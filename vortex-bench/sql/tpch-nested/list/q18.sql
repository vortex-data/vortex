select
    customer_data.c_name,
    c_custkey,
    o_orderkey,
    o_orderdate,
    orders_data.o_totalprice,
    sum(lineitem_data.l_quantity)
from
    (select c_custkey, c_nationkey, c_mktsegment, unnest(customer_data) as customer_data from customer) as customer,
    (select o_orderkey, o_custkey, o_orderdate, unnest(orders_data) as orders_data from orders) as orders,
    (select l_orderkey, l_partkey, l_suppkey, l_shipdate, unnest(lineitem_data) as lineitem_data from lineitem) as lineitem
where
    o_orderkey in (
        select
            l_orderkey
        from
            (select l_orderkey, l_partkey, l_suppkey, l_shipdate, unnest(lineitem_data) as lineitem_data from lineitem) as lineitem
        group by
            l_orderkey having
                sum(lineitem_data.l_quantity) > 300
    )
    and c_custkey = o_custkey
    and o_orderkey = l_orderkey
group by
    customer_data.c_name,
    c_custkey,
    o_orderkey,
    o_orderdate,
    orders_data.o_totalprice
order by
    orders_data.o_totalprice desc,
    o_orderdate
limit 100;
