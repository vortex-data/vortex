WITH RankedSuppliers AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        n.n_name AS nation_name,
        SUM(ps.ps_supplycost * ps.ps_availqty) AS total_supply_value,
        DENSE_RANK() OVER (PARTITION BY n.n_name ORDER BY SUM(ps.ps_supplycost * ps.ps_availqty) DESC) AS rank
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN 
        nation n ON s.s_nationkey = n.n_nationkey
    GROUP BY 
        s.s_suppkey, s.s_name, n.n_name
),
TopSuppliers AS (
    SELECT 
        rs.s_suppkey,
        rs.s_name,
        rs.nation_name,
        rs.total_supply_value
    FROM 
        RankedSuppliers rs
    WHERE 
        rs.rank <= 5
),
OrderDetails AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        li.l_partkey,
        li.l_quantity,
        li.l_discount,
        li.l_tax,
        ps.ps_supplycost,
        ts.s_name,
        ts.nation_name
    FROM 
        orders o
    JOIN 
        lineitem li ON o.o_orderkey = li.l_orderkey
    JOIN 
        TopSuppliers ts ON li.l_suppkey = ts.s_suppkey
    JOIN 
        partsupp ps ON li.l_partkey = ps.ps_partkey AND li.l_suppkey = ps.ps_suppkey
    WHERE 
        o.o_orderdate >= '1996-01-01' AND 
        o.o_orderdate < '1997-01-01'
)
SELECT 
    ts.nation_name,
    COUNT(DISTINCT od.o_orderkey) AS total_orders,
    SUM(od.l_quantity * od.ps_supplycost * (1 - od.l_discount)) AS total_revenue,
    AVG(od.l_tax) AS average_tax
FROM 
    OrderDetails od
JOIN 
    TopSuppliers ts ON od.s_name = ts.s_name
GROUP BY 
    ts.nation_name
ORDER BY 
    total_revenue DESC;