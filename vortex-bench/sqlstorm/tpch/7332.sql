
WITH part_summary AS (
    SELECT 
        p.p_partkey,
        p.p_name,
        SUM(ps.ps_availqty) AS total_available,
        AVG(ps.ps_supplycost) AS avg_supply_cost,
        SUM(l.l_quantity) AS total_quantity_ordered
    FROM 
        part p
    JOIN 
        partsupp ps ON p.p_partkey = ps.ps_partkey
    JOIN 
        lineitem l ON ps.ps_partkey = l.l_partkey
    GROUP BY 
        p.p_partkey, p.p_name
),
customer_orders AS (
    SELECT 
        c.c_custkey,
        c.c_name,
        COUNT(o.o_orderkey) AS order_count,
        SUM(o.o_totalprice) AS total_spent
    FROM 
        customer c
    LEFT JOIN 
        orders o ON c.c_custkey = o.o_custkey
    GROUP BY 
        c.c_custkey, c.c_name
),
nation_summary AS (
    SELECT 
        n.n_nationkey,
        n.n_name,
        SUM(c.total_spent) AS total_spent_by_nation,
        AVG(c.order_count) AS avg_orders_per_customer
    FROM 
        nation n
    LEFT JOIN 
        customer_orders c ON n.n_nationkey = c.c_custkey
    GROUP BY 
        n.n_nationkey, n.n_name
)
SELECT 
    ps.p_name,
    ps.total_available,
    ps.avg_supply_cost,
    ns.total_spent_by_nation,
    ns.avg_orders_per_customer
FROM 
    part_summary ps
JOIN 
    nation_summary ns ON ps.total_quantity_ordered > 1000
ORDER BY 
    ns.total_spent_by_nation DESC, ps.avg_supply_cost ASC;
