select
    sum(lineitem_data.l_extendedprice * (1 - lineitem_data.l_discount)) as revenue
from
    (select l_orderkey, l_partkey, l_suppkey, l_shipdate, unnest(lineitem_data) as lineitem_data from lineitem) as lineitem,
    (select p_partkey, p_brand, unnest(part_data) as part_data from part) as part
where
    (
        p_partkey = l_partkey
        and p_brand = 'Brand#12'
        and part_data.p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
        and lineitem_data.l_quantity >= 1 and lineitem_data.l_quantity <= 1 + 10
        and part_data.p_size between 1 and 5
        and lineitem_data.l_shipmode in ('AIR', 'AIR REG')
        and lineitem_data.l_shipinstruct = 'DELIVER IN PERSON'
    )
    or
    (
        p_partkey = l_partkey
        and p_brand = 'Brand#23'
        and part_data.p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
        and lineitem_data.l_quantity >= 10 and lineitem_data.l_quantity <= 10 + 10
        and part_data.p_size between 1 and 10
        and lineitem_data.l_shipmode in ('AIR', 'AIR REG')
        and lineitem_data.l_shipinstruct = 'DELIVER IN PERSON'
    )
    or
    (
        p_partkey = l_partkey
        and p_brand = 'Brand#34'
        and part_data.p_container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
        and lineitem_data.l_quantity >= 20 and lineitem_data.l_quantity <= 20 + 10
        and part_data.p_size between 1 and 15
        and lineitem_data.l_shipmode in ('AIR', 'AIR REG')
        and lineitem_data.l_shipinstruct = 'DELIVER IN PERSON'
    );
