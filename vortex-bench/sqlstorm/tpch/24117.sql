WITH RankedParts AS (
    SELECT 
        p.p_partkey,
        p.p_name,
        p.p_retailprice,
        ROW_NUMBER() OVER (PARTITION BY p.p_brand ORDER BY p.p_retailprice DESC) AS brand_rank
    FROM 
        part p
    WHERE 
        p.p_size BETWEEN 1 AND 30 
        AND p.p_retailprice IS NOT NULL
), 
CustomerOrders AS (
    SELECT 
        c.c_custkey,
        c.c_name,
        SUM(o.o_totalprice) AS total_spent,
        COUNT(o.o_orderkey) AS order_count
    FROM 
        customer c
    LEFT JOIN 
        orders o ON c.c_custkey = o.o_custkey
    GROUP BY 
        c.c_custkey, c.c_name
    HAVING 
        SUM(o.o_totalprice) IS NOT NULL
), 
SupplierInfo AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        SUM(ps.ps_supplycost * ps.ps_availqty) AS supplier_value
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY 
        s.s_suppkey, s.s_name
    HAVING 
        COUNT(ps.ps_partkey) > 0
)
SELECT 
    r.r_name,
    COALESCE(cp.c_name, 'Unknown Customer') AS top_customer,
    COALESCE(rp.p_name, 'No Ranked Parts') AS top_part_name,
    si.supplier_value,
    CASE 
        WHEN si.supplier_value >= 100000 THEN 'High Value'
        WHEN si.supplier_value BETWEEN 50000 AND 99999 THEN 'Medium Value'
        ELSE 'Low Value'
    END AS supplier_category
FROM 
    region r
LEFT JOIN 
    nation n ON r.r_regionkey = n.n_regionkey
LEFT JOIN 
    CustomerOrders cp ON n.n_nationkey = cp.c_custkey
LEFT JOIN 
    RankedParts rp ON rp.brand_rank = 1
LEFT JOIN 
    SupplierInfo si ON si.supplier_value >= 50000
WHERE 
    n.n_name LIKE '%land%' OR r.r_name IS NULL
ORDER BY 
    r.r_name, si.supplier_value DESC;
