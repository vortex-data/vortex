WITH ranked_orders AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        o.o_totalprice,
        RANK() OVER (PARTITION BY o.o_orderdate ORDER BY o.o_totalprice DESC) AS total_price_rank
    FROM 
        orders o
    WHERE 
        o.o_orderdate BETWEEN '1997-01-01' AND '1997-12-31'
),
top_suppliers AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        SUM(ps.ps_supplycost * ps.ps_availqty) AS total_supplycost
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY 
        s.s_suppkey, s.s_name
    ORDER BY 
        total_supplycost DESC
    LIMIT 10
),
customer_summary AS (
    SELECT 
        c.c_custkey,
        c.c_name,
        c.c_acctbal,
        COUNT(o.o_orderkey) AS order_count,
        SUM(o.o_totalprice) AS total_spent
    FROM 
        customer c
    LEFT JOIN 
        orders o ON c.c_custkey = o.o_custkey
    GROUP BY 
        c.c_custkey, c.c_name, c.c_acctbal
    HAVING 
        SUM(o.o_totalprice) > 10000
)
SELECT 
    r.o_orderkey,
    r.o_orderdate,
    cs.c_name,
    cs.total_spent,
    ts.s_name AS supplier_name,
    ts.total_supplycost
FROM 
    ranked_orders r
JOIN 
    customer_summary cs ON r.o_orderkey = cs.c_custkey
JOIN 
    lineitem l ON r.o_orderkey = l.l_orderkey
JOIN 
    top_suppliers ts ON l.l_suppkey = ts.s_suppkey
WHERE 
    r.total_price_rank <= 5
ORDER BY 
    r.o_orderdate, cs.total_spent DESC;