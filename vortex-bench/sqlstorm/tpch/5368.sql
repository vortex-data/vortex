WITH SupplierOrderCounts AS (
    SELECT s.s_suppkey, COUNT(DISTINCT o.o_orderkey) AS order_count
    FROM supplier s
    JOIN partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN lineitem l ON ps.ps_partkey = l.l_partkey
    JOIN orders o ON l.l_orderkey = o.o_orderkey
    WHERE o.o_orderstatus = 'O'
    GROUP BY s.s_suppkey
),
TopSuppliers AS (
    SELECT s.s_suppkey, s.s_name, soc.order_count
    FROM supplier s
    JOIN SupplierOrderCounts soc ON s.s_suppkey = soc.s_suppkey
    ORDER BY soc.order_count DESC
    LIMIT 5
),
PartDetails AS (
    SELECT p.p_partkey, p.p_name, p.p_mfgr, p.p_brand, p.p_retailprice
    FROM part p
    WHERE p.p_retailprice > (
        SELECT AVG(p2.p_retailprice)
        FROM part p2
    )
)
SELECT ts.s_name, ts.order_count, pd.p_name, pd.p_mfgr, pd.p_brand, pd.p_retailprice
FROM TopSuppliers ts
JOIN lineitem l ON ts.s_suppkey = l.l_suppkey
JOIN PartDetails pd ON l.l_partkey = pd.p_partkey
WHERE l.l_shipmode = 'REG AIR'
AND l.l_shipdate BETWEEN '1997-01-01' AND '1997-12-31'
ORDER BY ts.order_count DESC, pd.p_retailprice DESC;