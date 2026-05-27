WITH RankedSuppliers AS (
    SELECT 
        s.s_suppkey, 
        s.s_name, 
        SUM(ps.ps_supplycost * ps.ps_availqty) AS TotalCost,
        ROW_NUMBER() OVER (PARTITION BY n.n_name ORDER BY SUM(ps.ps_supplycost * ps.ps_availqty) DESC) AS SupplierRank
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN 
        nation n ON s.s_nationkey = n.n_nationkey
    GROUP BY 
        s.s_suppkey, s.s_name, n.n_name
), FilteredCustomers AS (
    SELECT 
        c.c_custkey, 
        c.c_name, 
        c.c_acctbal, 
        n.n_name, 
        n.n_nationkey
    FROM 
        customer c
    JOIN 
        nation n ON c.c_nationkey = n.n_nationkey
    WHERE 
        c.c_acctbal > (SELECT AVG(c_acctbal) FROM customer)
), OrderDetails AS (
    SELECT 
        o.o_orderkey, 
        o.o_orderdate, 
        o.o_totalprice, 
        l.l_partkey
    FROM 
        orders o
    JOIN 
        lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE 
        o.o_orderstatus = 'O' AND 
        l.l_shipdate BETWEEN '1996-01-01' AND '1996-12-31'
)
SELECT 
    fc.c_name AS CustomerName,
    fc.c_acctbal AS CustomerAccountBalance,
    rs.s_name AS SupplierName,
    rs.TotalCost AS SupplierTotalCost,
    od.o_orderkey AS OrderNumber,
    od.o_orderdate AS OrderDate,
    od.o_totalprice AS OrderTotalPrice
FROM 
    FilteredCustomers fc
JOIN 
    RankedSuppliers rs ON fc.n_nationkey = rs.s_suppkey
JOIN 
    OrderDetails od ON od.l_partkey IN (
        SELECT ps.ps_partkey 
        FROM partsupp ps 
        WHERE ps.ps_supplycost < 100.00
    )
WHERE 
    rs.SupplierRank <= 5 
ORDER BY 
    fc.c_name, rs.TotalCost DESC;