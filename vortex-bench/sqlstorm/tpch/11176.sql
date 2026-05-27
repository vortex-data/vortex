SELECT
    l_returnflag,
    l_linestatus,
    SUM(l_quantity) AS sum_quantity,
    SUM(l_extendedprice) AS sum_extendedprice,
    SUM(l_extendedprice * (1 - l_discount)) AS sum_discounted_price,
    AVG(l_tax) AS avg_tax,
    COUNT(*) AS total_lineitems
FROM
    lineitem
WHERE
    l_shipdate BETWEEN '1994-01-01' AND '1994-12-31'
GROUP BY
    l_returnflag,
    l_linestatus
ORDER BY
    l_returnflag,
    l_linestatus;
