select
    nation,
    o_year,
    sum(amount) as sum_profit
from
    (
        select
            nation_data.n_name as nation,
            extract(year from o_orderdate) as o_year,
            lineitem_data.l_extendedprice * (1 - lineitem_data.l_discount) - partsupp_data.ps_supplycost * lineitem_data.l_quantity as amount
        from
            (select p_partkey, p_brand, unnest(part_data) as part_data from part) as part,
            supplier,
            (select l_orderkey, l_partkey, l_suppkey, l_shipdate, unnest(lineitem_data) as lineitem_data from lineitem) as lineitem,
            (select ps_partkey, ps_suppkey, unnest(partsupp_data) as partsupp_data from partsupp) as partsupp,
            orders,
            (select n_nationkey, n_regionkey, unnest(nation_data) as nation_data from nation) as nation
        where
            s_suppkey = l_suppkey
            and ps_suppkey = l_suppkey
            and ps_partkey = l_partkey
            and p_partkey = l_partkey
            and o_orderkey = l_orderkey
            and s_nationkey = n_nationkey
            and part_data.p_name like '%green%'
    ) as profit
group by
    nation,
    o_year
order by
    nation,
    o_year desc;
