
SELECT 
    p.p_name,
    COUNT(DISTINCT ps.ps_suppkey) AS supplier_count,
    SUM(ps.ps_availqty) AS total_available_quantity,
    ROUND(AVG(ps.ps_supplycost), 2) AS average_supply_cost,
    SUBSTRING(p.p_comment, 1, 10) AS short_comment,
    CASE 
        WHEN CHAR_LENGTH(p.p_name) > 10 THEN 'Long Name' 
        ELSE 'Short Name' 
    END AS name_length_category
FROM 
    part p
JOIN 
    partsupp ps ON p.p_partkey = ps.ps_partkey
JOIN 
    supplier s ON ps.ps_suppkey = s.s_suppkey
WHERE 
    s.s_acctbal > 0 
    AND p.p_size BETWEEN 1 AND 30
GROUP BY 
    p.p_name, p.p_comment, p.p_size
HAVING 
    COUNT(DISTINCT ps.ps_suppkey) > 5
ORDER BY 
    total_available_quantity DESC
LIMIT 10;
