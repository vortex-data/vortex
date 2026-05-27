WITH RankedParts AS (
    SELECT 
        p.p_partkey,
        p.p_name,
        p.p_retailprice,
        ROW_NUMBER() OVER (PARTITION BY p.p_mfgr ORDER BY p.p_retailprice DESC) AS rnk
    FROM 
        part p
    WHERE 
        p.p_retailprice IS NOT NULL
), SupplierStats AS (
    SELECT 
        s.s_nationkey,
        COUNT(DISTINCT ps.ps_suppkey) AS supplier_count,
        AVG(s.s_acctbal) AS average_acctbal
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY 
        s.s_nationkey
), OrdersWithDiscount AS (
    SELECT 
        o.o_orderkey,
        SUM(l.l_extendedprice * (1 - l.l_discount)) AS total_price_with_discount
    FROM 
        orders o
    JOIN 
        lineitem l ON o.o_orderkey = l.l_orderkey
    GROUP BY 
        o.o_orderkey
), CombinedOrders AS (
    SELECT 
        o.o_orderkey,
        CASE 
            WHEN os.total_price_with_discount IS NOT NULL 
                THEN os.total_price_with_discount 
            ELSE 0 
        END AS adjusted_total_price
    FROM 
        OrdersWithDiscount os
    FULL OUTER JOIN orders o ON os.o_orderkey = o.o_orderkey
), FinalReport AS (
    SELECT 
        np.r_name,
        COALESCE(ss.supplier_count, 0) AS supplier_count,
        COALESCE(ss.average_acctbal, 0) AS average_acct_balance,
        COUNT(DISTINCT co.o_orderkey) AS total_orders,
        SUM(co.adjusted_total_price) AS total_revenue
    FROM 
        region np
    LEFT JOIN 
        nation n ON np.r_regionkey = n.n_regionkey
    LEFT JOIN 
        SupplierStats ss ON n.n_nationkey = ss.s_nationkey
    LEFT JOIN 
        CombinedOrders co ON n.n_nationkey = (CASE 
            WHEN ss.supplier_count > 0 THEN n.n_nationkey 
            ELSE NULL 
        END) 
    GROUP BY 
        np.r_name, ss.supplier_count, ss.average_acctbal
)
SELECT 
    fr.r_name,
    fr.supplier_count,
    fr.average_acct_balance,
    fr.total_orders,
    fr.total_revenue,
    STRING_AGG(DISTINCT rp.p_name, ', ') AS most_expensive_parts
FROM 
    FinalReport fr
LEFT JOIN 
    RankedParts rp ON fr.supplier_count > 0 AND fr.supplier_count IS NOT NULL
WHERE 
    fr.total_revenue > 10000
GROUP BY 
    fr.r_name, fr.supplier_count, fr.average_acct_balance, fr.total_orders, fr.total_revenue
HAVING 
    COUNT(rp.p_partkey) > 0
ORDER BY 
    fr.total_revenue DESC;
