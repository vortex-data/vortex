WITH RECURSIVE SupplierHierarchy AS (
    SELECT s_suppkey, s_name, s_acctbal, 0 AS level
    FROM supplier
    WHERE s_acctbal > 1000 
    UNION ALL
    SELECT s.s_suppkey, s.s_name, s.s_acctbal, sh.level + 1
    FROM supplier s
    JOIN SupplierHierarchy sh ON s.s_suppkey = sh.s_suppkey
),
TotalSales AS (
    SELECT c.c_custkey, SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_sales
    FROM customer c
    JOIN orders o ON c.c_custkey = o.o_custkey
    JOIN lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE o.o_orderstatus = 'O' AND l.l_shipdate >= '1997-01-01'
    GROUP BY c.c_custkey
),
HighValueCustomers AS (
    SELECT c.c_custkey, c.c_name, ts.total_sales
    FROM customer c
    JOIN TotalSales ts ON c.c_custkey = ts.c_custkey
    WHERE ts.total_sales > 5000
),
SupplierSales AS (
    SELECT s.s_suppkey, SUM(l.l_extendedprice * (1 - l.l_discount)) AS supplier_sales
    FROM supplier s
    JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN lineitem l ON ps.ps_partkey = l.l_partkey
    GROUP BY s.s_suppkey
),
FinalReport AS (
    SELECT 
        hvc.c_custkey,
        hvc.c_name,
        hvc.total_sales,
        ss.supplier_sales,
        CASE 
            WHEN ss.supplier_sales IS NULL THEN 'No sales'
            ELSE 'Sales exist'
        END AS sales_status
    FROM HighValueCustomers hvc
    LEFT JOIN SupplierSales ss ON hvc.c_custkey = ss.supplier_sales 
)

SELECT 
    fr.c_custkey,
    fr.c_name,
    fr.total_sales,
    COALESCE(fr.supplier_sales, 0) AS supplier_sales,
    fr.sales_status,
    ROW_NUMBER() OVER (ORDER BY fr.total_sales DESC) AS customer_rank
FROM FinalReport fr
ORDER BY fr.total_sales DESC, fr.c_name ASC;