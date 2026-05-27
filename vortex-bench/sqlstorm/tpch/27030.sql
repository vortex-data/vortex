SELECT 
    p.p_name, 
    COUNT(DISTINCT s.s_suppkey) AS supplier_count,
    AVG(ps.ps_supplycost) AS avg_supplycost,
    STRING_AGG(DISTINCT n.n_name, ', ') AS nations_supplied,
    RANK() OVER (ORDER BY AVG(ps.ps_supplycost) DESC) AS supply_rank
FROM 
    part p
JOIN 
    partsupp ps ON p.p_partkey = ps.ps_partkey
JOIN 
    supplier s ON ps.ps_suppkey = s.s_suppkey
JOIN 
    nation n ON s.s_nationkey = n.n_nationkey
WHERE 
    p.p_brand LIKE '%BrandA%' 
    AND LENGTH(p.p_comment) > 10 
    AND n.n_name NOT LIKE 'N%'
GROUP BY 
    p.p_name
HAVING 
    COUNT(DISTINCT s.s_suppkey) > 5
ORDER BY 
    supply_rank;
