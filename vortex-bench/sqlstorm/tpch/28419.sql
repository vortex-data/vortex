
WITH SupplierDetails AS (
    SELECT 
        s.s_suppkey,
        s.s_name,
        s.s_address,
        n.n_name AS nation_name,
        CONCAT(s.s_name, ' - ', s.s_address) AS supplier_info
    FROM 
        supplier s
    JOIN 
        nation n ON s.s_nationkey = n.n_nationkey
),
PartDetails AS (
    SELECT 
        p.p_partkey,
        p.p_name,
        p.p_mfgr,
        p.p_brand,
        p.p_retailprice,
        p.p_comment,
        LENGTH(p.p_comment) AS comment_length
    FROM 
        part p
),
CombinedData AS (
    SELECT 
        s.s_suppkey AS suppkey,
        s.supplier_info,
        p.p_name,
        p.p_brand,
        p.comment_length,
        p.p_retailprice
    FROM 
        SupplierDetails s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN 
        PartDetails p ON ps.ps_partkey = p.p_partkey
)
SELECT 
    cdc.supplier_info,
    cdc.p_name,
    cdc.p_brand,
    cdc.p_retailprice,
    COUNT(*) OVER (PARTITION BY cdc.supplier_info) AS supplier_part_count
FROM 
    CombinedData cdc
WHERE 
    cdc.comment_length > 20
GROUP BY 
    cdc.supplier_info,
    cdc.p_name,
    cdc.p_brand,
    cdc.p_retailprice,
    cdc.comment_length
ORDER BY 
    cdc.p_retailprice DESC, cdc.supplier_info;
