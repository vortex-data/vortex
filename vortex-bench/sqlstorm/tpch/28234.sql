SELECT 
    p.p_name,
    COUNT(DISTINCT ps.ps_suppkey) AS supplier_count,
    SUM(CASE WHEN o.o_orderstatus = 'F' THEN l.l_extendedprice ELSE 0 END) AS total_filled_order_value,
    STRING_AGG(DISTINCT s.s_name, ', ') AS supplier_names,
    SUBSTRING(p.p_comment, 1, 10) AS short_comment,
    CONCAT('Total Count: ', COUNT(DISTINCT ps.ps_suppkey), ' | Filled Value: ', 
           SUM(CASE WHEN o.o_orderstatus = 'F' THEN l.l_extendedprice ELSE 0 END) 
           ) AS report_summary
FROM 
    part p
JOIN 
    partsupp ps ON p.p_partkey = ps.ps_partkey
JOIN 
    supplier s ON ps.ps_suppkey = s.s_suppkey
JOIN 
    lineitem l ON ps.ps_partkey = l.l_partkey
JOIN 
    orders o ON l.l_orderkey = o.o_orderkey
WHERE 
    p.p_name LIKE 'widget%'
GROUP BY 
    p.p_name, p.p_comment
HAVING 
    COUNT(DISTINCT ps.ps_suppkey) > 0
ORDER BY 
    total_filled_order_value DESC;
