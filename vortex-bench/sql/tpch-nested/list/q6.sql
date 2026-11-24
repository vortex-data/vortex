select
    sum(lineitem_data.l_extendedprice * lineitem_data.l_discount) as revenue
from
    (select l_orderkey, l_partkey, l_suppkey, l_shipdate, unnest(lineitem_data) as lineitem_data from lineitem) as lineitem
where
    l_shipdate >= date '1994-01-01'
    and l_shipdate < date '1995-01-01'
    and lineitem_data.l_discount between 0.06 - 0.01 and 0.06 + 0.01
    and lineitem_data.l_quantity < 24;
