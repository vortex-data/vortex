WITH RankedSuppliers AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        n.n_name AS nation_name,
        SUM(ps.ps_supplycost * ps.ps_availqty) AS total_cost,
        RANK() OVER (PARTITION BY n.n_regionkey ORDER BY SUM(ps.ps_supplycost * ps.ps_availqty) DESC) AS supplier_rank
    FROM 
        supplier s
    JOIN 
        nation n ON s.s_nationkey = n.n_nationkey
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY 
        s.s_suppkey, s.s_name, n.n_name, n.n_regionkey
),
TopSuppliers AS (
    SELECT 
        r.r_name AS region_name,
        rs.s_name,
        rs.total_cost
    FROM 
        RankedSuppliers rs
    JOIN 
        nation n ON rs.nation_name = n.n_name
    JOIN 
        region r ON n.n_regionkey = r.r_regionkey
    WHERE 
        rs.supplier_rank <= 5
),
CustomerOrderSummary AS (
    SELECT 
        c.c_custkey,
        c.c_name,
        COUNT(o.o_orderkey) AS order_count,
        SUM(o.o_totalprice) AS total_spent
    FROM 
        customer c
    LEFT JOIN 
        orders o ON c.c_custkey = o.o_custkey
    WHERE 
        o.o_orderstatus = 'O'
    GROUP BY 
        c.c_custkey, c.c_name
)
SELECT 
    tos.region_name,
    tos.s_name,
    cos.c_name,
    cos.order_count,
    cos.total_spent
FROM 
    TopSuppliers tos
JOIN 
    CustomerOrderSummary cos ON tos.total_cost < cos.total_spent
ORDER BY 
    tos.region_name, tos.total_cost DESC, cos.total_spent DESC;
