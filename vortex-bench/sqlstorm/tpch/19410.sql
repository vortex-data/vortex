SELECT
    p_brand,
    COUNT(DISTINCT ps_suppkey) AS supplier_count,
    AVG(ps_supplycost) AS avg_supplycost
FROM
    part p
JOIN
    partsupp ps ON p.p_partkey = ps.ps_partkey
GROUP BY
    p_brand
ORDER BY
    supplier_count DESC
LIMIT 10;
