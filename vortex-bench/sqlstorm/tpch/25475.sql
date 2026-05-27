SELECT 
    r.r_name AS region_name,
    n.n_name AS nation_name,
    s.s_name AS supplier_name,
    COUNT(DISTINCT p.p_partkey) AS number_of_parts,
    SUM(ps.ps_availqty) AS total_available_quantity,
    AVG(s.s_acctbal) AS average_supplier_account_balance,
    STRING_AGG(DISTINCT p.p_type, ', ') AS unique_part_types,
    STRING_AGG(DISTINCT CONCAT(s.s_name, ' (', s.s_phone, ')'), '; ') AS suppliers_with_contact
FROM 
    region r
JOIN 
    nation n ON r.r_regionkey = n.n_regionkey
JOIN 
    supplier s ON n.n_nationkey = s.s_nationkey
JOIN 
    partsupp ps ON s.s_suppkey = ps.ps_suppkey
JOIN 
    part p ON ps.ps_partkey = p.p_partkey
WHERE 
    p.p_retailprice > 50.00
GROUP BY 
    r.r_name, n.n_name, s.s_name
HAVING 
    COUNT(DISTINCT p.p_partkey) > 10
ORDER BY 
    region_name ASC, nation_name ASC;
