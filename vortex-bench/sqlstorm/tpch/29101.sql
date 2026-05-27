SELECT 
    p.p_brand,
    COUNT(DISTINCT ps.ps_suppkey) AS supplier_count,
    SUM(ps.ps_availqty) AS total_available_quantity,
    AVG(ps.ps_supplycost) AS average_supply_cost,
    SUM(CASE 
        WHEN c.c_mktsegment = 'BUILDING' THEN l.l_extendedprice * (1 - l.l_discount)
        ELSE 0 
    END) AS building_segment_revenue,
    SUM(CASE 
        WHEN l.l_returnflag = 'R' THEN 1 
        ELSE 0 
    END) AS total_returns,
    STRING_AGG(DISTINCT r.r_name, ', ') AS regions_supplied
FROM 
    part p
JOIN 
    partsupp ps ON p.p_partkey = ps.ps_partkey
JOIN 
    supplier s ON ps.ps_suppkey = s.s_suppkey
JOIN 
    nation n ON s.s_nationkey = n.n_nationkey
JOIN 
    region r ON n.n_regionkey = r.r_regionkey
JOIN 
    lineitem l ON p.p_partkey = l.l_partkey
JOIN 
    orders o ON l.l_orderkey = o.o_orderkey
JOIN 
    customer c ON o.o_custkey = c.c_custkey
GROUP BY 
    p.p_brand
HAVING 
    COUNT(DISTINCT n.n_nationkey) > 1
ORDER BY 
    total_available_quantity DESC;
