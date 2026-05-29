
WITH ranked_customers AS (
    SELECT 
        c.c_custkey,
        c.c_name,
        c.c_acctbal,
        RANK() OVER (PARTITION BY c.c_nationkey ORDER BY c.c_acctbal DESC) AS rank,
        n.n_name
    FROM 
        customer c
    JOIN 
        nation n ON c.c_nationkey = n.n_nationkey
),
high_value_orders AS (
    SELECT 
        o.o_orderkey,
        o.o_custkey,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_value
    FROM 
        orders o
    JOIN 
        lineitem l ON o.o_orderkey = l.l_orderkey
    GROUP BY 
        o.o_orderkey, o.o_custkey
    HAVING 
        SUM(l.l_extendedprice * (1 - l.l_discount)) > 10000
),
supplier_part_info AS (
    SELECT 
        s.s_suppkey, 
        p.p_partkey,
        p.p_name,
        ps.ps_supplycost,
        ROW_NUMBER() OVER (PARTITION BY p.p_partkey ORDER BY ps.ps_supplycost DESC) AS supply_rank
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN 
        part p ON ps.ps_partkey = p.p_partkey
)
SELECT 
    r.n_name,
    rc.c_name,
    rc.c_acctbal,
    HO.o_orderkey,
    HO.total_value,
    STRING_AGG(DISTINCT CONCAT(sp.p_name, ': ', sp.ps_supplycost), ', ') AS part_supplier_info
FROM 
    ranked_customers rc
LEFT JOIN 
    high_value_orders HO ON rc.c_custkey = HO.o_custkey
LEFT JOIN 
    supplier_part_info sp ON rc.c_custkey = sp.s_suppkey 
LEFT JOIN 
    nation r ON rc.n_name = r.n_name
WHERE 
    rc.rank = 1
AND 
    HO.total_value IS NOT NULL
GROUP BY 
    r.n_name, rc.c_name, rc.c_acctbal, HO.o_orderkey, HO.total_value
ORDER BY 
    rc.c_acctbal DESC, r.n_name;
