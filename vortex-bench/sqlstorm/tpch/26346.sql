WITH RankedParts AS (
    SELECT 
        p.p_partkey, 
        p.p_name, 
        p.p_brand,
        p.p_type, 
        LENGTH(p.p_name) AS name_length,
        SUBSTRING(p.p_comment, 1, 10) AS short_comment
    FROM 
        part p
    WHERE 
        p.p_retailprice > 100.00
),
SupplierCounts AS (
    SELECT 
        ps.ps_partkey, 
        COUNT(DISTINCT ps.ps_suppkey) AS supplier_count
    FROM 
        partsupp ps
    GROUP BY 
        ps.ps_partkey
),
FinalResults AS (
    SELECT 
        r.p_partkey, 
        r.p_name, 
        r.p_brand, 
        r.p_type,
        s.supplier_count,
        CONCAT('Part: ', r.p_name, ' | Brand: ', r.p_brand) AS description
    FROM 
        RankedParts r
    JOIN 
        SupplierCounts s ON r.p_partkey = s.ps_partkey
)
SELECT 
    f.p_partkey,
    f.description,
    f.supplier_count
FROM 
    FinalResults f
WHERE 
    f.supplier_count > 2
ORDER BY 
    f.supplier_count DESC, 
    f.p_partkey;
