select
    100.00 * sum(case
                     when part_data.p_type like 'PROMO%'
                         then lineitem_data.l_extendedprice * (1 - lineitem_data.l_discount)
                     else 0
    end) / sum(lineitem_data.l_extendedprice * (1 - lineitem_data.l_discount)) as promo_revenue
from
    (select l_orderkey, l_partkey, l_suppkey, l_shipdate, unnest(lineitem_data) as lineitem_data from lineitem) as lineitem,
    (select p_partkey, p_brand, unnest(part_data) as part_data from part) as part
where
    l_partkey = p_partkey
    and l_shipdate >= date '1995-09-01'
    and l_shipdate < date '1995-10-01';
