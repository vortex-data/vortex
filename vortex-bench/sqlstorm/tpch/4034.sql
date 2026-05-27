
WITH RankedOrders AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        o.o_totalprice,
        o.o_custkey,
        ROW_NUMBER() OVER (PARTITION BY o.o_orderstatus ORDER BY o.o_totalprice DESC) AS rn
    FROM 
        orders o
    WHERE 
        o.o_orderdate >= DATE '1997-01-01' 
        AND o.o_orderdate < DATE '1998-01-01'
),
CustomerOrders AS (
    SELECT 
        c.c_custkey,
        c.c_name,
        SUM(o.o_totalprice) AS total_spent
    FROM 
        customer c
    JOIN 
        orders o ON c.c_custkey = o.o_custkey
    WHERE 
        c.c_acctbal > 1000
    GROUP BY 
        c.c_custkey, c.c_name
),
SupplierParts AS (
    SELECT 
        p.p_partkey,
        p.p_name,
        s.s_suppkey,
        s.s_name,
        ps.ps_availqty,
        ROW_NUMBER() OVER (PARTITION BY p.p_partkey ORDER BY ps.ps_supplycost DESC) AS rn
    FROM 
        part p
    JOIN 
        partsupp ps ON p.p_partkey = ps.ps_partkey
    JOIN 
        supplier s ON ps.ps_suppkey = s.s_suppkey
)
SELECT 
    co.c_name,
    rp.o_orderkey,
    rp.o_orderdate,
    COALESCE(SUM(l.l_extendedprice * (1 - l.l_discount)), 0) AS total_lineitem_price,
    sp.p_name,
    rp.rn AS rank_order_status,
    CASE 
        WHEN SUM(l.l_discount) > 0 THEN 'Discount Applied' 
        ELSE 'No Discount' 
    END AS discount_status
FROM 
    RankedOrders rp
LEFT JOIN 
    lineitem l ON rp.o_orderkey = l.l_orderkey
JOIN 
    CustomerOrders co ON rp.o_custkey = co.c_custkey
LEFT JOIN 
    SupplierParts sp ON l.l_partkey = sp.p_partkey AND sp.rn = 1
WHERE 
    rp.rn = 1
GROUP BY 
    co.c_name, rp.o_orderkey, rp.o_orderdate, sp.p_name, rp.rn
HAVING 
    COALESCE(SUM(l.l_extendedprice * (1 - l.l_discount)), 0) > 1000
ORDER BY 
    total_lineitem_price DESC;
