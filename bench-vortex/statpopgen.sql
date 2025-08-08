-- Count the number of variants.
SELECT COUNT(*) FROM statpopgen;
-- Count the number of samples (e.g. human beings) in this dataset. The GT arrays are all the same
-- length, even though they're not stored as a fixed-list type.
SELECT array_length("GT", 1) FROM statpopgen LIMIT 1;
-- Extract just the genotypes as a dense list of lists. Usually this is sent to a linear algebra
-- routine or statistical method.
SELECT "GT" FROM statpopgen;
-- Compute the frequency of the reference allele (all variants in this dataset are biallelic).
SELECT "CHROM", "POS", "REF", "ALT", 1.0 - CAST(LIST_SUM(GT) AS DOUBLE) / (2 * LIST_SUM(LIST_TRANSFORM(GT, lambda GT: GT IS NOT NULL))) AS reference_allele_frequency FROM statpopgen;
-- Compute the frequency of the alternate allele (all variants in this dataset are biallelic).
SELECT "CHROM", "POS", "REF", "ALT", CAST(LIST_SUM(GT) AS DOUBLE) / (2 * LIST_SUM(LIST_TRANSFORM(GT, lambda GT: GT IS NOT NULL))) AS alternate_allele_frequency FROM statpopgen;
-- Count the number of "called" (i.e. not null) genotypes per variant.
SELECT "CHROM", "POS", "REF", "ALT", LIST_SUM(LIST_TRANSFORM(GT, lambda GT: GT IS NOT NULL)) AS n_called FROM statpopgen;
-- Collect all the genotypes at variants whose minor allele frequency is < 10%.
SELECT "CHROM", "POS", "REF", "ALT", "GT",
       1.0 - CAST(LIST_SUM(GT) AS DOUBLE) / (2 * LIST_SUM(LIST_TRANSFORM(GT, lambda GT: GT IS NOT NULL))) AS reference_allele_frequency
  FROM statpopgen
 WHERE reference_allele_frequency > 0.9 OR reference_allele_frequency < 0.1;
-- Collect the necessary statistics for a Hardy-Weinberg Equalibrium test. The actual test involves
-- the Levene-Haldane distribution which is somewhat subtle to implement in SQL.
SELECT "CHROM", "POS", "REF", "ALT", "GT",
       LIST_SUM(CAST(GT == 0 AS DOUBLE)) as N_HOM_REF,
       LIST_SUM(CAST(GT == 1 AS DOUBLE)) as N_HET,
       LIST_SUM(CAST(GT == 2 AS DOUBLE)) as N_HOM_ALT
  FROM statpopgen;
-- Read just one variant (this is the sixth one).
SELECT *
  FROM statpopgen
 WHERE "CHROM" == "chr21"
   AND "POS" == 5030278;
-- Read a 700 base-pair window of variants.
SELECT *
  FROM statpopgen
 WHERE "CHROM" == "chr21"
   AND "POS" >= 5030300
   AND "POS" <= 5031000;
