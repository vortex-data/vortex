WITH RankedOrders AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        o.o_totalprice,
        ROW_NUMBER() OVER (PARTITION BY o.o_orderstatus ORDER BY o.o_totalprice DESC) AS rn
    FROM 
        orders o
    WHERE 
        o.o_orderdate >= DATE '1997-01-01' 
        AND o.o_orderdate <= DATE '1997-12-31'
),
SupplierParts AS (
    SELECT 
        ps.ps_partkey,
        ps.ps_suppkey,
        SUM(ps.ps_availqty) AS total_available,
        AVG(ps.ps_supplycost) AS avg_supplycost
    FROM 
        partsupp ps
    GROUP BY 
        ps.ps_partkey, ps.ps_suppkey
),
TopSuppliers AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_revenue,
        COUNT(DISTINCT l.l_orderkey) AS order_count
    FROM 
        supplier s
    JOIN 
        SupplierParts sp ON s.s_suppkey = sp.ps_suppkey
    JOIN 
        lineitem l ON l.l_partkey = sp.ps_partkey
    WHERE 
        l.l_shipdate IS NOT NULL
    GROUP BY 
        s.s_suppkey, s.s_name
    HAVING 
        SUM(l.l_extendedprice * (1 - l.l_discount)) > 100000
),
CustomerInfo AS (
    SELECT 
        c.c_custkey,
        c.c_name,
        COALESCE(SUM(o.o_totalprice), 0) AS total_spent,
        ROW_NUMBER() OVER (ORDER BY COALESCE(SUM(o.o_totalprice), 0) DESC) AS cust_rn
    FROM 
        customer c
    LEFT JOIN 
        orders o ON c.c_custkey = o.o_custkey
    GROUP BY 
        c.c_custkey, c.c_name
    HAVING 
        COUNT(o.o_orderkey) > 0
)
SELECT 
    r.r_name,
    COALESCE(c.c_name, 'Unknown Customer') AS customer_name,
    COALESCE(c.total_spent, 0) AS total_spent,
    t.total_revenue,
    t.order_count,
    CASE 
        WHEN t.total_revenue IS NOT NULL THEN (t.total_revenue / NULLIF(c.total_spent, 0)) * 100
        ELSE 0
    END AS revenue_ratio,
    DENSE_RANK() OVER (PARTITION BY r.r_name ORDER BY t.total_revenue DESC) AS revenue_rank
FROM 
    nation n 
LEFT JOIN 
    region r ON n.n_regionkey = r.r_regionkey
LEFT JOIN 
    CustomerInfo c ON n.n_nationkey = c.cust_rn
LEFT JOIN 
    TopSuppliers t ON n.n_nationkey = t.s_suppkey
WHERE 
    (c.total_spent > 5000 OR t.total_revenue > 20000) 
    AND (t.order_count > 10 OR c.cust_rn IS NULL)
ORDER BY 
    r.r_name, revenue_rank;