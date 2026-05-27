SELECT 
    n_name AS nation, 
    sum(l_extendedprice * (1 - l_discount)) AS total_revenue 
FROM 
    lineitem 
JOIN 
    orders ON l_orderkey = o_orderkey 
JOIN 
    customer ON o_custkey = c_custkey 
JOIN 
    nation ON c_nationkey = n_nationkey 
WHERE 
    l_shipdate >= '1997-01-01' AND l_shipdate < '1998-01-01' 
GROUP BY 
    n_name 
ORDER BY 
    total_revenue DESC;