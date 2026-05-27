WITH RankedOrders AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        o.o_totalprice,
        c.c_name,
        c.c_nationkey,
        ROW_NUMBER() OVER (PARTITION BY c.c_nationkey ORDER BY o.o_totalprice DESC) AS OrderRank
    FROM 
        orders o
    JOIN 
        customer c ON o.o_custkey = c.c_custkey
    WHERE 
        o.o_orderdate >= DATE '1997-01-01'
        AND o.o_orderdate < DATE '1997-10-01'
),
HighValueOrders AS (
    SELECT 
        ro.o_orderkey,
        ro.o_orderdate,
        ro.o_totalprice,
        ro.c_name,
        n.n_name AS nation_name
    FROM 
        RankedOrders ro
    JOIN 
        nation n ON ro.c_nationkey = n.n_nationkey
    WHERE 
        ro.OrderRank <= 10
)
SELECT 
    hvo.o_orderkey,
    hvo.o_orderdate,
    hvo.o_totalprice,
    hvo.c_name,
    hvo.nation_name,
    SUM(li.l_quantity) AS total_quantity,
    AVG(li.l_extendedprice) AS avg_extended_price,
    SUM(li.l_discount) AS total_discount
FROM 
    HighValueOrders hvo
JOIN 
    lineitem li ON hvo.o_orderkey = li.l_orderkey
GROUP BY 
    hvo.o_orderkey, hvo.o_orderdate, hvo.o_totalprice, hvo.c_name, hvo.nation_name
ORDER BY 
    hvo.o_totalprice DESC;