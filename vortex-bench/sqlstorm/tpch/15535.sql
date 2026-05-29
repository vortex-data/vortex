SELECT 
    p_brand, 
    COUNT(p_partkey) AS part_count, 
    AVG(p_retailprice) AS avg_price 
FROM 
    part 
GROUP BY 
    p_brand 
HAVING 
    COUNT(p_partkey) > 10 
ORDER BY 
    avg_price DESC;
