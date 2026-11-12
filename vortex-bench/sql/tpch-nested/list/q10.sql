select
    c_custkey,
    customer_data.c_name,
    sum(lineitem_data.l_extendedprice * (1 - lineitem_data.l_discount)) as revenue,
    customer_data.c_acctbal,
    nation_data.n_name,
    customer_data.c_address,
    customer_data.c_phone,
    customer_data.c_comment
from
    (select c_custkey, c_nationkey, c_mktsegment, unnest(customer_data) as customer_data from customer) as customer,
    orders,
    (select l_orderkey, l_partkey, l_suppkey, l_shipdate, unnest(lineitem_data) as lineitem_data from lineitem) as lineitem,
    (select n_nationkey, n_regionkey, unnest(nation_data) as nation_data from nation) as nation
where
    c_custkey = o_custkey
    and l_orderkey = o_orderkey
    and o_orderdate >= date '1993-10-01'
    and o_orderdate < date '1994-01-01'
    and lineitem_data.l_returnflag = 'R'
    and c_nationkey = n_nationkey
group by
    c_custkey,
    customer_data.c_name,
    customer_data.c_acctbal,
    customer_data.c_phone,
    nation_data.n_name,
    customer_data.c_address,
    customer_data.c_comment
order by
    revenue desc
limit 20;
