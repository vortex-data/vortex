WITH NationalSales AS (
    SELECT 
        n.n_name AS nation,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_sales,
        COUNT(DISTINCT o.o_orderkey) AS order_count
    FROM 
        nation n
    JOIN 
        supplier s ON n.n_nationkey = s.s_nationkey
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN 
        part p ON ps.ps_partkey = p.p_partkey
    JOIN 
        lineitem l ON p.p_partkey = l.l_partkey
    JOIN 
        orders o ON l.l_orderkey = o.o_orderkey
    WHERE 
        o.o_orderstatus = 'O'
    GROUP BY 
        n.n_name
),
AverageSales AS (
    SELECT 
        AVG(total_sales) AS avg_sales
    FROM 
        NationalSales
)
SELECT 
    ns.nation,
    ns.total_sales,
    ns.order_count,
    CASE 
        WHEN ns.total_sales > (SELECT avg_sales FROM AverageSales) THEN 'Above Average'
        ELSE 'Below Average'
    END AS sales_category
FROM 
    NationalSales ns
ORDER BY 
    ns.total_sales DESC;