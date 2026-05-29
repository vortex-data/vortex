WITH RankedSuppliers AS (
    SELECT 
        s.s_suppkey, 
        s.s_name, 
        s.s_acctbal, 
        s.s_nationkey,
        ROW_NUMBER() OVER (PARTITION BY s.s_nationkey ORDER BY s.s_acctbal DESC) as rn
    FROM 
        supplier s
),
AvailableParts AS (
    SELECT 
        ps.ps_partkey, 
        SUM(ps.ps_availqty) AS total_available
    FROM 
        partsupp ps
    GROUP BY 
        ps.ps_partkey
),
HighValueOrders AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_value
    FROM 
        orders o
    JOIN 
        lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE 
        o.o_orderstatus = 'O'
    GROUP BY 
        o.o_orderkey, o.o_orderdate
    HAVING 
        SUM(l.l_extendedprice * (1 - l.l_discount)) > 10000
)
SELECT 
    n.n_name, 
    p.p_name, 
    COALESCE(SUM(hvo.total_value), 0) AS total_order_value,
    AVG(COALESCE(rs.s_acctbal, 0)) AS avg_supplier_balance,
    COUNT(DISTINCT rs.s_suppkey) AS supplier_count,
    pot.total_available,
    CASE 
        WHEN COUNT(DISTINCT rs.s_suppkey) > 0 THEN 'Suppliers Available'
        ELSE 'No Suppliers Available' 
    END AS supplier_availability
FROM 
    part p
LEFT JOIN 
    partsupp ps ON p.p_partkey = ps.ps_partkey
LEFT JOIN 
    RankedSuppliers rs ON rs.s_suppkey = ps.ps_suppkey AND rs.rn <= 5
LEFT JOIN 
    AvailableParts pot ON pot.ps_partkey = p.p_partkey
LEFT JOIN 
    HighValueOrders hvo ON hvo.o_orderkey = ps.ps_partkey
JOIN 
    nation n ON n.n_nationkey = rs.s_nationkey
WHERE 
    p.p_size = (
        SELECT MAX(p2.p_size) 
        FROM part p2 
        WHERE p2.p_type = p.p_type
    )
GROUP BY 
    n.n_name, p.p_name, pot.total_available
ORDER BY 
    total_order_value DESC, avg_supplier_balance DESC;
