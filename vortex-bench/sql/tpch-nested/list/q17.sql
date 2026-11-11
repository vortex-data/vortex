select
    sum(lineitem_data.l_extendedprice) / 7.0 as avg_yearly
from
    (select l_orderkey, l_partkey, l_suppkey, l_shipdate, unnest(lineitem_data) as lineitem_data from lineitem) as lineitem,
    (select p_partkey, p_brand, unnest(part_data) as part_data from part) as part
where
    p_partkey = l_partkey
    and p_brand = 'Brand#23'
    and part_data.p_container = 'MED BOX'
    and lineitem_data.l_quantity < (
        select
            0.2 * avg(lineitem_data.l_quantity)
        from
            (select l_orderkey, l_partkey, l_suppkey, l_shipdate, unnest(lineitem_data) as lineitem_data from lineitem) as lineitem
        where
            l_partkey = p_partkey
    );
