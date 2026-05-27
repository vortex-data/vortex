WITH RankedSuppliers AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        SUM(COALESCE(ps.ps_availqty, 0)) AS total_available_qty,
        COUNT(DISTINCT p.p_partkey) AS part_count,
        ROW_NUMBER() OVER (PARTITION BY r.r_regionkey ORDER BY SUM(COALESCE(ps.ps_supplycost, 0)) DESC) AS rn
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN 
        part p ON ps.ps_partkey = p.p_partkey
    JOIN 
        nation n ON s.s_nationkey = n.n_nationkey
    JOIN 
        region r ON n.n_regionkey = r.r_regionkey
    WHERE 
        p.p_comment LIKE '%special%'
    GROUP BY 
        s.s_suppkey, s.s_name, r.r_regionkey
),
TopSuppliers AS (
    SELECT 
        rs.s_suppkey,
        rs.s_name,
        rs.total_available_qty,
        rs.part_count
    FROM 
        RankedSuppliers rs
    WHERE 
        rs.rn <= 3
)
SELECT 
    ts.s_name,
    ts.total_available_qty,
    ts.part_count,
    (SELECT COUNT(DISTINCT c.c_custkey) 
     FROM customer c 
     JOIN orders o ON c.c_custkey = o.o_custkey 
     JOIN lineitem l ON o.o_orderkey = l.l_orderkey 
     WHERE l.l_suppkey = ts.s_suppkey) AS total_customers
FROM 
    TopSuppliers ts
ORDER BY 
    ts.total_available_qty DESC;
