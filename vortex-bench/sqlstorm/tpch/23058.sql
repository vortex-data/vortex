
WITH RankedSuppliers AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        s.s_acctbal,
        RANK() OVER (PARTITION BY n.n_nationkey ORDER BY s.s_acctbal DESC) AS rank
    FROM 
        supplier s 
    JOIN 
        nation n ON s.s_nationkey = n.n_nationkey
),
HighValueParts AS (
    SELECT 
        p.p_partkey,
        p.p_name,
        SUM(ps.ps_availqty * ps.ps_supplycost) AS total_parts_value
    FROM 
        part p 
    JOIN 
        partsupp ps ON p.p_partkey = ps.ps_partkey
    GROUP BY 
        p.p_partkey, p.p_name
    HAVING 
        SUM(ps.ps_availqty) > 1000
),
OrderDetails AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS order_total
    FROM 
        orders o
    JOIN 
        lineitem l ON o.o_orderkey = l.l_orderkey
    GROUP BY 
        o.o_orderkey, o.o_orderdate
    HAVING 
        SUM(l.l_discount) IS NULL OR SUM(l.l_discount) < 0.10
),
FinalReport AS (
    SELECT 
        ns.n_name AS nation_name,
        rs.s_name AS supplier_name,
        hp.p_name AS part_name,
        od.order_total,
        ROW_NUMBER() OVER (PARTITION BY ns.n_nationkey ORDER BY od.order_total DESC) AS order_rank
    FROM 
        RankedSuppliers rs
    JOIN 
        nation ns ON rs.s_suppkey = ns.n_nationkey
    JOIN 
        HighValueParts hp ON hp.total_parts_value > 0
    JOIN 
        OrderDetails od ON od.o_orderkey IN (
            SELECT o.o_orderkey
            FROM orders o
            WHERE o.o_orderstatus = 'O' AND o.o_totalprice < 5000
        )
    LEFT JOIN 
        lineitem l ON od.o_orderkey = l.l_orderkey AND l.l_returnflag = 'R'
    WHERE 
        rs.rank = 1
)
SELECT 
    fr.nation_name,
    fr.supplier_name,
    fr.part_name,
    fr.order_total,
    CASE 
        WHEN fr.order_rank IS NULL THEN 'No Rank Available' 
        ELSE CAST(fr.order_rank AS VARCHAR)
    END AS order_rank
FROM 
    FinalReport fr
WHERE 
    fr.order_total IS NOT NULL
ORDER BY 
    fr.nation_name, fr.order_total DESC;
