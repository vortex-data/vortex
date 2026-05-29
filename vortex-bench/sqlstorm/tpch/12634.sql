SELECT 
    l_orderkey, 
    SUM(l_extendedprice * (1 - l_discount)) AS total_revenue
FROM 
    lineitem
WHERE 
    l_shipdate >= '1996-01-01' AND l_shipdate < '1996-02-01'
GROUP BY 
    l_orderkey
ORDER BY 
    total_revenue DESC
LIMIT 10;