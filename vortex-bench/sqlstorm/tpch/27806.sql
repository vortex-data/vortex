WITH String_Benchmark AS (
    SELECT 
        p.p_name,
        LENGTH(p.p_name) AS name_length,
        UPPER(p.p_mfgr) AS upper_mfgr,
        LOWER(p.p_brand) AS lower_brand,
        CONCAT(p.p_type, ' - ', p.p_container) AS type_container,
        REPLACE(p.p_comment, 'smokeless', 'flame-free') AS modified_comment,
        SUBSTRING(p.p_comment, 1, 10) AS comment_excerpt
    FROM part p
    WHERE p.p_retailprice > 100.00
)
SELECT 
    sb.name_length,
    COUNT(sb.upper_mfgr) AS upper_mfgr_count,
    COUNT(sb.lower_brand) AS lower_brand_count,
    COUNT(sb.type_container) AS type_container_count,
    COUNT(sb.modified_comment) AS modified_comment_count,
    COUNT(sb.comment_excerpt) AS comment_excerpt_count
FROM String_Benchmark sb
GROUP BY sb.name_length
ORDER BY sb.name_length DESC;
