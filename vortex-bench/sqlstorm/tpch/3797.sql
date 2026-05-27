WITH RegionalSales AS (
    SELECT 
        n.n_name AS nation,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_sales,
        ROW_NUMBER() OVER (PARTITION BY n.n_nationkey ORDER BY SUM(l.l_extendedprice * (1 - l.l_discount)) DESC) AS sales_rank
    FROM 
        lineitem l
    JOIN 
        orders o ON l.l_orderkey = o.o_orderkey
    JOIN 
        customer c ON o.o_custkey = c.c_custkey
    JOIN 
        nation n ON c.c_nationkey = n.n_nationkey
    WHERE 
        o.o_orderdate >= DATE '1996-01-01' AND o.o_orderdate < DATE '1997-01-01'
    GROUP BY 
        n.n_nationkey, n.n_name
),
TopNations AS (
    SELECT 
        nation,
        total_sales
    FROM 
        RegionalSales
    WHERE 
        sales_rank <= 5
),
SupplierCosts AS (
    SELECT 
        s.s_name AS supplier_name,
        SUM(ps.ps_supplycost * ps.ps_availqty) AS total_cost
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN 
        part p ON ps.ps_partkey = p.p_partkey
    WHERE 
        p.p_size > 20
    GROUP BY 
        s.s_name
)
SELECT 
    t.nation,
    t.total_sales,
    COALESCE(s.total_cost, 0) AS supplier_total_cost,
    CASE 
        WHEN t.total_sales > 1.0 * COALESCE(s.total_cost, 0) THEN 'Profitable'
        ELSE 'Not Profitable'
    END AS profitability_status
FROM 
    TopNations t
LEFT JOIN 
    SupplierCosts s ON t.nation = s.supplier_name
ORDER BY 
    t.total_sales DESC, supplier_total_cost ASC;