WITH SupplierSales AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_sales
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN 
        lineitem l ON ps.ps_partkey = l.l_partkey
    GROUP BY 
        s.s_suppkey, s.s_name
),
CustomerOrders AS (
    SELECT 
        c.c_custkey,
        c.c_name,
        SUM(o.o_totalprice) AS total_order_value
    FROM 
        customer c
    JOIN 
        orders o ON c.c_custkey = o.o_custkey
    WHERE 
        o.o_orderdate >= DATE '1994-01-01' 
        AND o.o_orderdate < DATE '1995-01-01'
    GROUP BY 
        c.c_custkey, c.c_name
),
NationSummary AS (
    SELECT 
        n.n_nationkey,
        n.n_name,
        COUNT(DISTINCT s.s_suppkey) AS supplier_count
    FROM 
        nation n
    LEFT JOIN 
        supplier s ON n.n_nationkey = s.s_nationkey
    GROUP BY 
        n.n_nationkey, n.n_name
),
FinalReport AS (
    SELECT 
        cs.c_name,
        cs.total_order_value,
        ss.total_sales,
        ns.supplier_count,
        RANK() OVER (ORDER BY cs.total_order_value DESC) AS order_rank
    FROM 
        CustomerOrders cs
    LEFT JOIN 
        SupplierSales ss ON cs.c_custkey = ss.s_suppkey
    LEFT JOIN 
        NationSummary ns ON cs.c_custkey = ns.n_nationkey
)

SELECT 
    fr.c_name,
    fr.total_order_value,
    COALESCE(fr.total_sales, 0) AS total_sales,
    fr.supplier_count,
    CASE 
        WHEN fr.order_rank <= 10 THEN 'Top Customer'
        ELSE 'Regular Customer'
    END AS customer_category
FROM 
    FinalReport fr
WHERE 
    fr.total_order_value > 10000 OR fr.total_sales > 50000
ORDER BY 
    fr.total_order_value DESC;