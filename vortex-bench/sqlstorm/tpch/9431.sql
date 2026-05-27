WITH SupplierParts AS (
    SELECT 
        s.s_suppkey, 
        s.s_name, 
        SUM(ps.ps_supplycost * ps.ps_availqty) AS total_cost
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY 
        s.s_suppkey, s.s_name
),
HighCostSuppliers AS (
    SELECT 
        s.s_suppkey, 
        s.s_name
    FROM 
        SupplierParts s
    WHERE 
        s.total_cost > (SELECT AVG(total_cost) FROM SupplierParts)
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
)
SELECT 
    co.c_name AS customer_name,
    h.s_name AS supplier_name,
    co.total_spent AS total_customer_spent,
    sp.total_cost AS total_supplier_cost
FROM 
    HighCostSuppliers h
JOIN 
    SupplierParts sp ON h.s_suppkey = sp.s_suppkey
JOIN 
    CustomerOrders co ON co.total_spent > sp.total_cost
ORDER BY 
    co.total_spent DESC, sp.total_cost DESC
LIMIT 10;
