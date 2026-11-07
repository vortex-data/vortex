select
    supplier_data.s_name,
    supplier_data.s_address
from
    (select s_suppkey, s_nationkey, unnest(supplier_data) as supplier_data from supplier) as supplier,
    (select n_nationkey, n_regionkey, unnest(nation_data) as nation_data from nation) as nation
where
    s_suppkey in (
        select
            ps_suppkey
        from
            (select ps_partkey, ps_suppkey, unnest(partsupp_data) as partsupp_data from partsupp) as partsupp
        where
            ps_partkey in (
                select
                    p_partkey
                from
                    (select p_partkey, p_brand, unnest(part_data) as part_data from part) as part
                where
                    part_data.p_name like 'forest%'
            )
            and partsupp_data.ps_availqty > (
                select
                    0.5 * sum(lineitem_data.l_quantity)
                from
                    (select l_orderkey, l_partkey, l_suppkey, l_shipdate, unnest(lineitem_data) as lineitem_data from lineitem) as lineitem
                where
                    l_partkey = ps_partkey
                    and l_suppkey = ps_suppkey
                    and l_shipdate >= date '1994-01-01'
                    and l_shipdate < date '1994-01-01' + interval '1' year
            )
    )
    and s_nationkey = n_nationkey
    and nation_data.n_name = 'CANADA'
order by
    supplier_data.s_name;
