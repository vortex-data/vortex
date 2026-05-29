WITH RECURSIVE MonthlyOrders AS (
    SELECT 
        o_custkey,
        DATE_TRUNC('month', o_orderdate) AS order_month,
        COUNT(o_orderkey) AS order_count,
        SUM(o_totalprice) AS total_revenue
    FROM 
        orders
    WHERE 
        o_orderdate >= DATE '1997-01-01' AND o_orderdate < DATE '1997-12-31'
    GROUP BY 
        o_custkey, DATE_TRUNC('month', o_orderdate)
),
NationSummary AS (
    SELECT
        n.n_nationkey,
        n.n_name,
        COUNT(DISTINCT s.s_suppkey) AS total_suppliers,
        SUM(ps.ps_supplycost * ps.ps_availqty) AS total_supply_value
    FROM
        nation n
    LEFT JOIN
        supplier s ON n.n_nationkey = s.s_nationkey
    LEFT JOIN
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY
        n.n_nationkey, n.n_name
),
HighValueCustomers AS (
    SELECT 
        c.c_custkey,
        c.c_name,
        SUM(mo.total_revenue) AS high_value_revenue
    FROM
        customer c
    JOIN
        MonthlyOrders mo ON c.c_custkey = mo.o_custkey
    WHERE
        mo.order_count > 5
    GROUP BY
        c.c_custkey, c.c_name
)
SELECT 
    n.n_name,
    COALESCE(hvc.c_name, 'No High Value Customers') AS high_value_customer_name,
    n.total_suppliers,
    n.total_supply_value,
    ROW_NUMBER() OVER (PARTITION BY n.n_nationkey ORDER BY n.total_supply_value DESC) AS rank
FROM 
    NationSummary n
LEFT JOIN 
    HighValueCustomers hvc ON hvc.high_value_revenue > 5000
ORDER BY 
    n.n_name, rank;