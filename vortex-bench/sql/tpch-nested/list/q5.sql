select
    nation_data.n_name,
    sum(lineitem_data.l_extendedprice * (1 - lineitem_data.l_discount)) as revenue
from
    customer,
    orders,
    (select l_orderkey, l_partkey, l_suppkey, l_shipdate, unnest(lineitem_data) as lineitem_data from lineitem) as lineitem,
    supplier,
    (select n_nationkey, n_regionkey, unnest(nation_data) as nation_data from nation) as nation,
    (select r_regionkey, unnest(region_data) as region_data from region) as region
where
    c_custkey = o_custkey
    and l_orderkey = o_orderkey
    and l_suppkey = s_suppkey
    and c_nationkey = s_nationkey
    and s_nationkey = n_nationkey
    and n_regionkey = r_regionkey
    and region_data.r_name = 'ASIA'
    and o_orderdate >= date '1994-01-01'
    and o_orderdate < date '1995-01-01'
group by
    nation_data.n_name
order by
    revenue desc;
