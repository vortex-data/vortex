SELECT 
    CONCAT_WS(' ', c.c_name, s.s_name) AS supplier_customer_name,
    LEFT(p.p_name, 15) AS short_part_name,
    COUNT(DISTINCT o.o_orderkey) AS total_orders,
    SUM(l.l_quantity) AS total_quantity,
    AVG(l.l_extendedprice * (1 - l.l_discount)) AS avg_price_after_discount,
    MAX(l.l_shipdate) AS last_ship_date,
    MIN(l.l_shipdate) AS first_ship_date,
    SUM(CASE WHEN l.l_returnflag = 'R' THEN 1 ELSE 0 END) AS total_returns,
    SUM(CASE WHEN l.l_linestatus = 'F' THEN l.l_quantity ELSE 0 END) AS fulfilled_quantity
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
    part p ON ps.ps_partkey = p.p_partkey
WHERE 
    p.p_comment LIKE '%fragile%'
    AND c.c_mktsegment = 'BUILDING'
GROUP BY 
    c.c_name, s.s_name, p.p_name
ORDER BY 
    total_orders DESC, avg_price_after_discount DESC;
