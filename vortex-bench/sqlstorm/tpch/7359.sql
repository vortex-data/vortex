WITH RankedOrders AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        o.o_totalprice,
        c.c_name,
        ROW_NUMBER() OVER (PARTITION BY c.c_nationkey ORDER BY o.o_orderdate DESC) AS order_rank
    FROM 
        orders o
    JOIN 
        customer c ON o.o_custkey = c.c_custkey
),
RecentHighValueOrders AS (
    SELECT 
        ro.o_orderkey,
        ro.o_orderdate,
        ro.o_totalprice,
        ro.c_name
    FROM 
        RankedOrders ro
    WHERE 
        ro.order_rank <= 5 
        AND ro.o_totalprice > (SELECT AVG(o_totalprice) FROM orders)
),
SupplierDetails AS (
    SELECT 
        ps.ps_partkey,
        ps.ps_suppkey,
        s.s_name,
        s.s_acctbal
    FROM 
        partsupp ps
    JOIN 
        supplier s ON ps.ps_suppkey = s.s_suppkey
)
SELECT 
    rhv.o_orderkey,
    rhv.o_orderdate,
    rhv.o_totalprice,
    rhv.c_name,
    sd.s_name AS supplier_name,
    sd.s_acctbal AS supplier_account_balance
FROM 
    RecentHighValueOrders rhv
JOIN 
    lineitem li ON rhv.o_orderkey = li.l_orderkey
JOIN 
    SupplierDetails sd ON li.l_partkey = sd.ps_partkey
ORDER BY 
    rhv.o_orderdate DESC, rhv.o_totalprice DESC;
