SELECT 
    n.n_name, 
    SUM(ps.ps_supplycost * l.l_quantity) AS total_cost
FROM 
    lineitem l
JOIN 
    partsupp ps ON l.l_partkey = ps.ps_partkey
JOIN 
    supplier s ON ps.ps_suppkey = s.s_suppkey
JOIN 
    nation n ON s.s_nationkey = n.n_nationkey
GROUP BY 
    n.n_name
ORDER BY 
    total_cost DESC
LIMIT 10;