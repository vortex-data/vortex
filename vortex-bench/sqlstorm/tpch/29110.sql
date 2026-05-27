SELECT 
    CONCAT(c.c_name, ' from ', s.s_name, ' in ', n.n_name, ' supplies ', p.p_name) AS supply_info,
    LENGTH(CONCAT(c.c_name, ' from ', s.s_name, ' in ', n.n_name, ' supplies ', p.p_name)) AS info_length,
    SUBSTRING(CONCAT(c.c_name, ' from ', s.s_name, ' in ', n.n_name, ' supplies ', p.p_name), 1, 50) AS short_supply_info
FROM 
    customer c
JOIN 
    orders o ON c.c_custkey = o.o_custkey
JOIN 
    lineitem l ON o.o_orderkey = l.l_orderkey
JOIN 
    partsupp ps ON l.l_partkey = ps.ps_partkey
JOIN 
    supplier s ON ps.ps_suppkey = s.s_suppkey
JOIN 
    nation n ON s.s_nationkey = n.n_nationkey
JOIN 
    part p ON ps.ps_partkey = p.p_partkey
WHERE 
    LENGTH(s.s_comment) > 50 
    AND o.o_orderstatus = 'O'
ORDER BY 
    info_length DESC
LIMIT 10;
