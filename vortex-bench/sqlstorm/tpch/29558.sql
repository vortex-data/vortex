WITH StringMetrics AS (
    SELECT 
        p.p_partkey,
        p.p_name,
        LENGTH(p.p_name) AS name_length,
        SUBSTRING(p.p_name FROM 1 FOR 3) AS name_prefix,
        REPLACE(p.p_comment, ' ', '') AS compact_comment,
        COUNT(DISTINCT s.s_name) AS supplier_count,
        SUM(ps.ps_availqty) AS total_available
    FROM 
        part p
    LEFT JOIN 
        partsupp ps ON p.p_partkey = ps.ps_partkey
    LEFT JOIN 
        supplier s ON ps.ps_suppkey = s.s_suppkey
    GROUP BY 
        p.p_partkey, p.p_name, p.p_comment
),
OrderedMetrics AS (
    SELECT 
        sm.p_partkey,
        sm.p_name,
        sm.name_length,
        sm.name_prefix,
        sm.compact_comment,
        sm.supplier_count,
        sm.total_available,
        ROW_NUMBER() OVER (ORDER BY sm.supplier_count DESC, sm.total_available DESC) AS rank
    FROM 
        StringMetrics sm
)
SELECT 
    om.p_partkey,
    om.p_name,
    om.name_length,
    om.name_prefix,
    om.compact_comment,
    om.supplier_count,
    om.total_available,
    CONCAT('Rank: ', om.rank) AS rank_string
FROM 
    OrderedMetrics om
WHERE 
    om.supplier_count > 0
ORDER BY 
    om.rank;
