
SELECT 
    CONCAT('Supplier: ', s.s_name, ' | Nation: ', n.n_name, 
           ' | Region: ', r.r_name, ' | Total Order Value: ', 
           SUM(l.l_extendedprice * (1 - l.l_discount))) AS Total_Value, 
    s.s_name, n.n_name, r.r_name
FROM 
    supplier s 
JOIN 
    nation n ON s.s_nationkey = n.n_nationkey 
JOIN 
    region r ON n.n_regionkey = r.r_regionkey 
JOIN 
    partsupp ps ON s.s_suppkey = ps.ps_suppkey 
JOIN 
    part p ON ps.ps_partkey = p.p_partkey 
JOIN 
    lineitem l ON p.p_partkey = l.l_partkey 
JOIN 
    orders o ON l.l_orderkey = o.o_orderkey 
WHERE 
    o.o_orderdate BETWEEN '1997-01-01' AND '1997-12-31' 
GROUP BY 
    s.s_name, n.n_name, r.r_name 
HAVING 
    SUM(l.l_extendedprice * (1 - l.l_discount)) > 10000 
ORDER BY 
    Total_Value DESC;
