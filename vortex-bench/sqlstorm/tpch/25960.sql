WITH SupplierDetails AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        s.s_address,
        n.n_name AS nation_name,
        COUNT(DISTINCT ps.ps_partkey) AS total_parts,
        SUM(ps.ps_availqty) AS total_available_qty,
        SUM(ps.ps_supplycost) AS total_supply_cost
    FROM 
        supplier s
    JOIN 
        nation n ON s.s_nationkey = n.n_nationkey
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    GROUP BY 
        s.s_suppkey, s.s_name, s.s_address, n.n_name
),
ProductStats AS (
    SELECT 
        p.p_partkey,
        p.p_name,
        p.p_brand,
        p.p_type,
        p.p_size,
        SUM(l.l_quantity) AS total_quantity,
        SUM(l.l_extendedprice) AS total_revenue,
        SUM(l.l_discount) AS total_discount,
        AVG(p.p_retailprice) AS avg_retail_price
    FROM 
        part p
    JOIN 
        lineitem l ON p.p_partkey = l.l_partkey
    GROUP BY 
        p.p_partkey, p.p_name, p.p_brand, p.p_type, p.p_size
),
BenchmarkResults AS (
    SELECT 
        sd.s_suppkey,
        sd.s_name,
        sd.nation_name,
        ps.p_partkey,
        ps.p_name,
        ps.total_quantity,
        ps.total_revenue,
        ps.avg_retail_price,
        sd.total_parts,
        sd.total_available_qty,
        sd.total_supply_cost
    FROM 
        SupplierDetails sd
    JOIN 
        ProductStats ps ON sd.total_parts > 50 AND sd.total_available_qty > 100
    ORDER BY 
        sd.total_supply_cost DESC, ps.total_revenue DESC
)
SELECT 
    BR.s_suppkey,
    BR.s_name,
    BR.nation_name,
    BR.p_partkey,
    BR.p_name,
    BR.total_quantity,
    BR.total_revenue,
    BR.avg_retail_price,
    BR.total_parts,
    BR.total_available_qty,
    BR.total_supply_cost
FROM 
    BenchmarkResults BR
WHERE 
    BR.total_revenue > 100000
LIMIT 100;
