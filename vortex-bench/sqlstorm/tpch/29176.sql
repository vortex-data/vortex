SELECT 
    p.p_name,
    COUNT(DISTINCT ps.ps_suppkey) AS supplier_count,
    AVG(s.s_acctbal) AS avg_supplier_acctbal,
    SUM(l.l_quantity) AS total_ordered_quantity,
    STRING_AGG(DISTINCT s.s_comment, '; ') AS supplier_comments,
    MAX(CASE 
            WHEN l.l_returnflag = 'R' THEN l.l_extendedprice * (1 - l.l_discount)
            ELSE 0 
        END) AS max_returned_value
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
    o.o_orderdate BETWEEN '1997-01-01' AND '1997-12-31' 
    AND p.p_type LIKE '%rubber%'
GROUP BY 
    p.p_name
ORDER BY 
    supplier_count DESC, 
    total_ordered_quantity DESC
LIMIT 10;