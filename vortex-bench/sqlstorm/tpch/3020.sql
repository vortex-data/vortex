WITH OrderDetails AS (
    SELECT 
        o.o_orderkey,
        o.o_orderdate,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_sales,
        COUNT(DISTINCT l.l_linenumber) AS line_item_count
    FROM 
        orders o
    JOIN 
        lineitem l ON o.o_orderkey = l.l_orderkey
    WHERE 
        o.o_orderdate >= DATE '1996-01-01' AND o.o_orderdate < DATE '1997-01-01'
    GROUP BY 
        o.o_orderkey, o.o_orderdate
),
RankedOrders AS (
    SELECT 
        od.o_orderkey,
        od.o_orderdate,
        od.total_sales,
        od.line_item_count,
        RANK() OVER (ORDER BY od.total_sales DESC) AS sales_rank
    FROM 
        OrderDetails od
)
SELECT 
    coalesce(r.o_orderkey, o.o_orderkey) as order_key,
    o.o_orderdate,
    r.total_sales,
    r.line_item_count,
    CASE 
        WHEN r.sales_rank IS NULL THEN 'No Sales'
        ELSE CONCAT('Rank ', r.sales_rank)
    END AS sales_rank
FROM 
    orders o
LEFT JOIN 
    RankedOrders r ON o.o_orderkey = r.o_orderkey
WHERE 
    o.o_orderstatus IN ('O', 'P') 
    AND (r.total_sales IS NULL OR r.total_sales > 1000) 
    AND EXISTS (
        SELECT 1 
        FROM lineitem l 
        WHERE l.l_orderkey = o.o_orderkey 
        AND l.l_returnflag = 'R'
    )
ORDER BY 
    o.o_orderdate DESC, total_sales DESC;