WITH RankedOrders AS (
    SELECT 
        o.o_orderkey, 
        o.o_orderdate,
        o.o_totalprice,
        RANK() OVER (PARTITION BY o.o_orderdate ORDER BY o.o_totalprice DESC) AS OrderRank
    FROM 
        orders o
    WHERE 
        o.o_orderstatus = 'O' AND 
        o.o_orderdate >= DATE '1996-01-01'
),
SupplierStats AS (
    SELECT 
        s.s_suppkey,
        SUM(ps.ps_availqty) AS TotalAvailableQty,
        AVG(ps.ps_supplycost) AS AvgSupplyCost
    FROM 
        supplier s
        JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY 
        s.s_suppkey
),
TopSuppliers AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        ss.TotalAvailableQty,
        ss.AvgSupplyCost
    FROM 
        supplier s
        JOIN SupplierStats ss ON s.s_suppkey = ss.s_suppkey
    WHERE 
        ss.TotalAvailableQty > 1000
)
SELECT 
    ro.o_orderkey,
    ro.o_orderdate,
    ro.o_totalprice,
    ts.s_name AS SupplierName,
    ts.TotalAvailableQty,
    ts.AvgSupplyCost
FROM 
    RankedOrders ro
LEFT JOIN 
    lineitem l ON ro.o_orderkey = l.l_orderkey
LEFT JOIN 
    TopSuppliers ts ON l.l_suppkey = ts.s_suppkey
WHERE 
    (ro.OrderRank <= 5 OR ts.AvgSupplyCost IS NULL)
ORDER BY 
    ro.o_orderdate DESC, 
    ro.o_totalprice DESC;