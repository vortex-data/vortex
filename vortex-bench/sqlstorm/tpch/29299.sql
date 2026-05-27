WITH RankedSuppliers AS (
    SELECT 
        s.s_suppkey, 
        s.s_name, 
        s.s_acctbal, 
        ROW_NUMBER() OVER (PARTITION BY p.p_partkey ORDER BY s.s_acctbal DESC) AS rn,
        p.p_name, 
        p.p_container
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN 
        part p ON ps.ps_partkey = p.p_partkey
    WHERE 
        p.p_size BETWEEN 10 AND 20 
        AND s.s_comment LIKE '%reliable%'
), FilteredOrders AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        o.o_totalprice,
        RANK() OVER (ORDER BY o.o_totalprice DESC) AS order_rank
    FROM 
        orders o
    WHERE 
        o.o_orderdate BETWEEN '1997-01-01' AND '1997-12-31'
        AND o.o_orderstatus = 'F'
)
SELECT 
    fs.s_name, 
    fs.p_name, 
    fs.p_container, 
    fo.o_orderkey, 
    fo.o_orderdate, 
    fo.o_totalprice
FROM 
    RankedSuppliers fs
JOIN 
    FilteredOrders fo ON fs.rn = 1
ORDER BY 
    fo.o_totalprice DESC, 
    fs.p_name ASC;