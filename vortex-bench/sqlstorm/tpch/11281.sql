SELECT 
    p.p_partkey, 
    p.p_name, 
    s.s_name AS supplier_name, 
    ps.ps_supplycost, 
    ps.ps_availqty, 
    SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue 
FROM 
    part p 
JOIN 
    partsupp ps ON p.p_partkey = ps.ps_partkey 
JOIN 
    supplier s ON ps.ps_suppkey = s.s_suppkey 
JOIN 
    lineitem l ON p.p_partkey = l.l_partkey 
JOIN 
    orders o ON l.l_orderkey = o.o_orderkey 
WHERE 
    o.o_orderdate >= '1997-01-01' AND o.o_orderdate < '1997-12-31' 
GROUP BY 
    p.p_partkey, p.p_name, s.s_name, ps.ps_supplycost, ps.ps_availqty 
ORDER BY 
    total_revenue DESC 
LIMIT 10;