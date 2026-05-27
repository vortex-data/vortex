SELECT 
    CONCAT('Supplier: ', s_name, ' | Part: ', p_name, ' | Price: $', p_retailprice) AS product_details,
    c_name AS customer_name,
    o_orderdate,
    COUNT(DISTINCT o_orderkey) AS total_orders,
    SUM(l_quantity) AS total_quantity,
    AVG(l_extendedprice) AS avg_price
FROM 
    supplier s
JOIN 
    partsupp ps ON s.s_suppkey = ps.ps_suppkey
JOIN 
    part p ON ps.ps_partkey = p.p_partkey
JOIN 
    lineitem l ON l.l_partkey = p.p_partkey
JOIN 
    orders o ON l.l_orderkey = o.o_orderkey
JOIN 
    customer c ON o.o_custkey = c.c_custkey
WHERE 
    s_name LIKE 'Supplier%'
    AND o_orderdate BETWEEN '1997-01-01' AND '1997-12-31'
GROUP BY 
    product_details, c_name, o_orderdate
ORDER BY 
    total_quantity DESC, avg_price ASC;