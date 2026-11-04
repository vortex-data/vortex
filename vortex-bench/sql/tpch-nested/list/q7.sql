select
    supp_nation,
    cust_nation,
    l_year,
    sum(volume) as revenue
from
    (
        select
            n1.nation_data.n_name as supp_nation,
            n2.nation_data.n_name as cust_nation,
            extract(year from l_shipdate) as l_year,
            lineitem_data.l_extendedprice * (1 - lineitem_data.l_discount) as volume
        from
            supplier,
            (select l_orderkey, l_partkey, l_suppkey, l_shipdate, unnest(lineitem_data) as lineitem_data from lineitem) as lineitem,
            orders,
            customer,
            (select n_nationkey, n_regionkey, unnest(nation_data) as nation_data from nation) as n1,
            (select n_nationkey, n_regionkey, unnest(nation_data) as nation_data from nation) as n2
        where
            s_suppkey = l_suppkey
            and o_orderkey = l_orderkey
            and c_custkey = o_custkey
            and s_nationkey = n1.n_nationkey
            and c_nationkey = n2.n_nationkey
            and (
                (n1.nation_data.n_name = 'FRANCE' and n2.nation_data.n_name = 'GERMANY')
                or (n1.nation_data.n_name = 'GERMANY' and n2.nation_data.n_name = 'FRANCE')
            )
            and l_shipdate between date '1995-01-01' and date '1996-12-31'
    ) as shipping
group by
    supp_nation,
    cust_nation,
    l_year
order by
    supp_nation,
    cust_nation,
    l_year;
