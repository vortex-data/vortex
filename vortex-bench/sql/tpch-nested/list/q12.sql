select
    lineitem_data.l_shipmode,
    sum(case
            when orders_data.o_orderpriority = '1-URGENT'
                or orders_data.o_orderpriority = '2-HIGH'
                then 1
            else 0
        end) as high_line_count,
    sum(case
            when orders_data.o_orderpriority <> '1-URGENT'
                and orders_data.o_orderpriority <> '2-HIGH'
                then 1
            else 0
        end) as low_line_count
from
    (select l_orderkey, l_partkey, l_suppkey, l_shipdate, unnest(lineitem_data) as lineitem_data from lineitem) as lineitem
        join
    (select o_orderkey, o_custkey, o_orderdate, unnest(orders_data) as orders_data from orders) as orders
    on
        l_orderkey = o_orderkey
where
    lineitem_data.l_shipmode in ('MAIL', 'SHIP')
    and lineitem_data.l_commitdate < lineitem_data.l_receiptdate
    and l_shipdate < lineitem_data.l_commitdate
    and lineitem_data.l_receiptdate >= date '1994-01-01'
    and lineitem_data.l_receiptdate < date '1995-01-01'
group by
    lineitem_data.l_shipmode
order by
    lineitem_data.l_shipmode;
