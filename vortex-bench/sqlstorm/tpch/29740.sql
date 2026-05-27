
WITH FilteredParts AS (
    SELECT p.p_partkey, p.p_name, p.p_retailprice, p.p_comment
    FROM part p
    WHERE p.p_retailprice > 50.00
      AND LENGTH(p.p_comment) > 10
), SupplierDetails AS (
    SELECT s.s_suppkey, s.s_name, s.s_acctbal, s.s_comment
    FROM supplier s
    WHERE s.s_acctbal < 1000.00
), CombinedData AS (
    SELECT fp.p_partkey, fp.p_name, fp.p_retailprice, sd.s_name AS supplier_name, sd.s_acctbal
    FROM FilteredParts fp
    JOIN partsupp ps ON fp.p_partkey = ps.ps_partkey
    JOIN SupplierDetails sd ON ps.ps_suppkey = sd.s_suppkey
)
SELECT COUNT(*) AS total_parts,
       MIN(p_retailprice) AS min_retail_price,
       MAX(p_retailprice) AS max_retail_price,
       AVG(p_retailprice) AS avg_retail_price,
       STRING_AGG(DISTINCT supplier_name, ', ') AS supplier_names
FROM CombinedData
WHERE p_retailprice BETWEEN 50.00 AND 200.00
GROUP BY p_partkey, p_name, p_retailprice
ORDER BY total_parts DESC;
