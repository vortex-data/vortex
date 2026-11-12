select
    p_brand,
    part_data.p_type,
    part_data.p_size,
    count(distinct ps_suppkey) as supplier_cnt
from
    partsupp,
    (select p_partkey, p_brand, unnest(part_data) as part_data from part) as part
where
    p_partkey = ps_partkey
    and p_brand <> 'Brand#45'
    and part_data.p_type not like 'MEDIUM POLISHED%'
    and part_data.p_size in (49, 14, 23, 45, 19, 3, 36, 9)
    and ps_suppkey not in (
        select
            s_suppkey
        from
            (select s_suppkey, s_nationkey, unnest(supplier_data) as supplier_data from supplier) as supplier
        where
            supplier_data.s_comment like '%Customer%Complaints%'
    )
group by
    p_brand,
    part_data.p_type,
    part_data.p_size
order by
    supplier_cnt desc,
    p_brand,
    part_data.p_type,
    part_data.p_size;
