select
    supplier_data.s_acctbal,
    supplier_data.s_name,
    nation_data.n_name,
    p_partkey,
    part_data.p_mfgr,
    supplier_data.s_address,
    supplier_data.s_phone,
    supplier_data.s_comment
from
    (select p_partkey, p_brand, unnest(part_data) as part_data from part) as part,
    (select s_suppkey, s_nationkey, unnest(supplier_data) as supplier_data from supplier) as supplier,
    (select ps_partkey, ps_suppkey, unnest(partsupp_data) as partsupp_data from partsupp) as partsupp,
    (select n_nationkey, n_regionkey, unnest(nation_data) as nation_data from nation) as nation,
    (select r_regionkey, unnest(region_data) as region_data from region) as region
where
    p_partkey = ps_partkey
    and s_suppkey = ps_suppkey
    and part_data.p_size = 15
    and part_data.p_type like '%BRASS'
    and s_nationkey = n_nationkey
    and n_regionkey = r_regionkey
    and region_data.r_name = 'EUROPE'
    and partsupp_data.ps_supplycost = (
        select
            min(partsupp_data.ps_supplycost)
        from
            (select ps_partkey, ps_suppkey, unnest(partsupp_data) as partsupp_data from partsupp) as partsupp,
            supplier,
            nation,
            (select r_regionkey, unnest(region_data) as region_data from region) as region
        where
            p_partkey = ps_partkey
            and s_suppkey = ps_suppkey
            and s_nationkey = n_nationkey
            and n_regionkey = r_regionkey
            and region_data.r_name = 'EUROPE'
    )
order by
    supplier_data.s_acctbal desc,
    nation_data.n_name,
    supplier_data.s_name,
    p_partkey
limit 100;
