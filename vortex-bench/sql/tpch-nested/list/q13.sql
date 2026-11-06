select
    c_count,
    count(*) as custdist
from
    (
        select
            c_custkey,
            count(o_orderkey)
        from
            customer left outer join (select o_orderkey, o_custkey, o_orderdate, unnest(orders_data) as orders_data from orders) as orders on
                c_custkey = o_custkey
                and orders_data.o_comment not like '%special%requests%'
        group by
            c_custkey
    ) as c_orders (c_custkey, c_count)
group by
    c_count
order by
    custdist desc,
    c_count desc;
