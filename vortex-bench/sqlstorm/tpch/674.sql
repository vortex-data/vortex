WITH RankedSuppliers AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        SUM(ps.ps_supplycost * ps.ps_availqty) AS total_cost,
        ROW_NUMBER() OVER (PARTITION BY n.n_name ORDER BY SUM(ps.ps_supplycost * ps.ps_availqty) DESC) AS rank
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN 
        nation n ON s.s_nationkey = n.n_nationkey
    GROUP BY 
        s.s_suppkey, s.s_name, n.n_name
),
CustomerOrders AS (
    SELECT 
        c.c_custkey,
        c.c_name,
        COUNT(o.o_orderkey) AS total_orders,
        SUM(o.o_totalprice) AS total_spent
    FROM 
        customer c
    LEFT JOIN 
        orders o ON c.c_custkey = o.o_custkey
    GROUP BY 
        c.c_custkey, c.c_name
),
HighValueCustomers AS (
    SELECT 
        c.c_custkey,
        c.c_name,
        co.total_orders,
        co.total_spent
    FROM 
        CustomerOrders co
    JOIN 
        customer c ON co.c_custkey = c.c_custkey
    WHERE 
        co.total_spent > 10000
)
SELECT 
    n.n_name,
    COUNT(DISTINCT cu.c_custkey) AS high_value_customers_count,
    AVG(s.total_cost) AS average_supplier_cost
FROM 
    HighValueCustomers cu
LEFT JOIN 
    RankedSuppliers s ON s.rank = 1
JOIN 
    nation n ON cu.c_custkey = n.n_nationkey
GROUP BY 
    n.n_name
HAVING 
    COUNT(DISTINCT cu.c_custkey) > 0
ORDER BY 
    n.n_name;
