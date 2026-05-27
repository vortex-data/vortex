WITH OrderSummary AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue,
        COUNT(DISTINCT l.l_suppkey) AS unique_suppliers,
        C.c_mktsegment
    FROM 
        orders o
    JOIN 
        lineitem l ON o.o_orderkey = l.l_orderkey
    JOIN 
        customer C ON o.o_custkey = C.c_custkey
    WHERE 
        o.o_orderdate >= DATE '1995-01-01' 
        AND o.o_orderdate < DATE '1996-01-01'
    GROUP BY 
        o.o_orderkey, o.o_orderdate, C.c_mktsegment
),
SupplierPerformance AS (
    SELECT 
        ps.ps_suppkey,
        SUM(ps.ps_supplycost * ps.ps_availqty) AS supplier_costs,
        COUNT(DISTINCT ps.ps_partkey) AS supplied_parts
    FROM 
        partsupp ps
    JOIN 
        supplier s ON ps.ps_suppkey = s.s_suppkey
    GROUP BY 
        ps.ps_suppkey
)
SELECT 
    O.o_orderkey,
    O.o_orderdate,
    O.total_revenue,
    O.unique_suppliers,
    S.supplier_costs,
    S.supplied_parts,
    O.c_mktsegment
FROM 
    OrderSummary O
JOIN 
    SupplierPerformance S ON O.unique_suppliers = S.supplied_parts
WHERE 
    O.total_revenue > (SELECT AVG(total_revenue) FROM OrderSummary)
ORDER BY 
    O.total_revenue DESC;
