SELECT
    p.p_partkey,
    p.p_name,
    s.s_suppkey,
    s.s_name,
    SUM(ps.ps_availqty) AS total_available_quantity,
    SUM(ps.ps_supplycost) AS total_supply_cost
FROM
    part p
JOIN
    partsupp ps ON p.p_partkey = ps.ps_partkey
JOIN
    supplier s ON ps.ps_suppkey = s.s_suppkey
GROUP BY
    p.p_partkey, p.p_name, s.s_suppkey, s.s_name
ORDER BY
    total_available_quantity DESC, total_supply_cost DESC
LIMIT 100;
