WITH RankedOrders AS (
    SELECT 
        o.o_orderkey,
        o.o_totalprice,
        CUME_DIST() OVER (PARTITION BY o.o_orderstatus ORDER BY o.o_totalprice DESC) AS price_rank,
        o.o_orderdate,
        c.c_nationkey,
        n.n_name
    FROM 
        orders o
    JOIN 
        customer c ON o.o_custkey = c.c_custkey
    JOIN 
        nation n ON c.c_nationkey = n.n_nationkey
    WHERE 
        o.o_orderstatus IN ('O', 'F')
),
SupplierStats AS (
    SELECT 
        s.s_suppkey,
        SUM(ps.ps_supplycost * ps.ps_availqty) AS total_supply_cost,
        COUNT(DISTINCT p.p_partkey) AS part_count
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN 
        part p ON ps.ps_partkey = p.p_partkey
    GROUP BY 
        s.s_suppkey
),
FilteredSuppliers AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        ss.total_supply_cost,
        CASE 
            WHEN ss.part_count > 5 THEN 'High Supplier'
            ELSE 'Low Supplier' 
        END AS supplier_type
    FROM 
        supplier s
    JOIN 
        SupplierStats ss ON s.s_suppkey = ss.s_suppkey
    WHERE 
        ss.total_supply_cost > (SELECT AVG(total_supply_cost) FROM SupplierStats) OR s.s_name LIKE 'A%'
),
FinalReport AS (
    SELECT 
        ro.o_orderkey,
        ro.o_totalprice,
        ro.o_orderdate,
        fs.s_name,
        fs.supplier_type,
        ro.n_name,
        COALESCE(SUM(l.l_extendedprice * (1 - l.l_discount)), 0) AS net_revenue
    FROM 
        RankedOrders ro
    LEFT JOIN 
        lineitem l ON ro.o_orderkey = l.l_orderkey
    LEFT JOIN 
        FilteredSuppliers fs ON l.l_suppkey = fs.s_suppkey
    GROUP BY 
        ro.o_orderkey, ro.o_totalprice, ro.o_orderdate, fs.s_name, fs.supplier_type, ro.n_name
)
SELECT 
    f.n_name,
    SUM(f.net_revenue) AS total_net_revenue,
    COUNT(DISTINCT f.o_orderkey) AS order_count,
    MAX(f.o_totalprice) AS highest_order_price,
    COUNT(CASE WHEN f.supplier_type = 'High Supplier' THEN 1 END) AS high_supplier_count
FROM 
    FinalReport f
WHERE 
    f.net_revenue > 0
GROUP BY 
    f.n_name
ORDER BY 
    total_net_revenue DESC
LIMIT 10;
