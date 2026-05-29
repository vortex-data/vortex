SELECT 
    p.p_partkey, 
    SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue
FROM 
    part p
JOIN 
    lineitem l ON p.p_partkey = l.l_partkey
WHERE 
    l.l_shipdate >= DATE '1997-01-01' AND l.l_shipdate < DATE '1997-12-31'
GROUP BY 
    p.p_partkey
ORDER BY 
    total_revenue DESC
LIMIT 10;