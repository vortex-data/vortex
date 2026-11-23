select
    lineitem_data.l_returnflag,
    lineitem_data.l_linestatus,
    sum(lineitem_data.l_quantity) as sum_qty,
    sum(lineitem_data.l_extendedprice) as sum_base_price,
    sum(lineitem_data.l_extendedprice * (1 - lineitem_data.l_discount)) as sum_disc_price,
    sum(lineitem_data.l_extendedprice * (1 - lineitem_data.l_discount) * (1 + lineitem_data.l_tax)) as sum_charge,
    avg(lineitem_data.l_quantity) as avg_qty,
    avg(lineitem_data.l_extendedprice) as avg_price,
    avg(lineitem_data.l_discount) as avg_disc,
    count(*) as count_order
from
    (select l_orderkey, l_partkey, l_suppkey, l_shipdate, unnest(lineitem_data) as lineitem_data from lineitem) as lineitem
where
    l_shipdate <= date '1998-09-02'
group by
    lineitem_data.l_returnflag,
    lineitem_data.l_linestatus
order by
    lineitem_data.l_returnflag,
    lineitem_data.l_linestatus;
