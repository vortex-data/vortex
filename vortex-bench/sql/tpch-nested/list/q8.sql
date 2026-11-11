select
    o_year,
    sum(case
            when nation = 'BRAZIL' then volume
            else 0
        end) / sum(volume) as mkt_share
from
    (
        select
            extract(year from o_orderdate) as o_year,
            lineitem_data.l_extendedprice * (1 - lineitem_data.l_discount) as volume,
            n2.nation_data.n_name as nation
        from
            (select p_partkey, p_brand, unnest(part_data) as part_data from part) as part,
            supplier,
            (select l_orderkey, l_partkey, l_suppkey, l_shipdate, unnest(lineitem_data) as lineitem_data from lineitem) as lineitem,
            orders,
            customer,
            (select n_nationkey, n_regionkey, unnest(nation_data) as nation_data from nation) as n1,
            (select n_nationkey, n_regionkey, unnest(nation_data) as nation_data from nation) as n2,
            (select r_regionkey, unnest(region_data) as region_data from region) as region
        where
            p_partkey = l_partkey
            and s_suppkey = l_suppkey
            and l_orderkey = o_orderkey
            and o_custkey = c_custkey
            and c_nationkey = n1.n_nationkey
            and n1.n_regionkey = r_regionkey
            and region_data.r_name = 'AMERICA'
            and s_nationkey = n2.n_nationkey
            and o_orderdate between date '1995-01-01' and date '1996-12-31'
            and part_data.p_type = 'ECONOMY ANODIZED STEEL'
    ) as all_nations
group by
    o_year
order by
    o_year;
