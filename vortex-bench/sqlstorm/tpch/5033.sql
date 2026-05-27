WITH NationwideSales AS (
    SELECT 
        n.n_name AS nation,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_sales
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
        n.n_name
), 
SalesBySupplier AS (
    SELECT 
        s.s_name AS supplier_name,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS supplier_sales
    FROM 
        lineitem l
    JOIN 
        partsupp ps ON l.l_partkey = ps.ps_partkey
    JOIN 
        supplier s ON ps.ps_suppkey = s.s_suppkey
    GROUP BY 
        s.s_name
), 
TopSellingParts AS (
    SELECT 
        p.p_name AS part_name,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS part_sales
    FROM 
        lineitem l
    JOIN 
        partsupp ps ON l.l_partkey = ps.ps_partkey
    JOIN 
        part p ON ps.ps_partkey = p.p_partkey
    GROUP BY 
        p.p_name
    ORDER BY 
        part_sales DESC
    LIMIT 10
)

SELECT 
    ns.nation, 
    ns.total_sales, 
    ss.supplier_name, 
    ss.supplier_sales, 
    tp.part_name, 
    tp.part_sales
FROM 
    NationwideSales ns
JOIN 
    SalesBySupplier ss ON ss.supplier_sales > 100000
JOIN 
    TopSellingParts tp ON tp.part_sales > 50000
ORDER BY 
    ns.total_sales DESC, 
    ss.supplier_sales DESC, 
    tp.part_sales DESC
LIMIT 50;