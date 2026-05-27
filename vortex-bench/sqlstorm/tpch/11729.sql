SELECT
    n.n_name,
    SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue
FROM
    part p
JOIN
    lineitem l ON p.p_partkey = l.l_partkey
JOIN
    partsupp ps ON ps.ps_partkey = p.p_partkey
JOIN
    supplier s ON s.s_suppkey = ps.ps_suppkey
JOIN
    nation n ON n.n_nationkey = s.s_nationkey
GROUP BY
    n.n_name
ORDER BY
    total_revenue DESC
LIMIT 10;
