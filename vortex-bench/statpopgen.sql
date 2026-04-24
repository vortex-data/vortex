-- 0. Count the number of variants.
SELECT COUNT(*) FROM statpopgen;
-- 1.
--
-- Count the number of samples (e.g. human beings) in this dataset. The GT arrays are all the same
-- length, even though they're not stored as a fixed-list type.
SELECT array_length("GT", 1) FROM statpopgen LIMIT 1;
-- 2.
--
-- Extract just the genotypes as a dense list of lists. Usually this is sent to a linear algebra
-- routine or statistical method.
SELECT "GT" FROM statpopgen;
-- 3. Compute the frequency of the reference allele (all variants in this dataset are biallelic).
SELECT "CHROM", "POS", "REF", "ALT", 1.0 - CAST(LIST_SUM(GT) AS DOUBLE) / (2 * LIST_SUM(LIST_TRANSFORM(GT, lambda GT: GT IS NOT NULL))) AS reference_allele_frequency FROM statpopgen;
-- 4. Compute the frequency of the alternate allele (all variants in this dataset are biallelic).
SELECT "CHROM", "POS", "REF", "ALT", CAST(LIST_SUM(GT) AS DOUBLE) / (2 * LIST_SUM(LIST_TRANSFORM(GT, lambda GT: GT IS NOT NULL))) AS alternate_allele_frequency FROM statpopgen;
-- 5. Count the number of "called" (i.e. not null) genotypes per variant.
SELECT "CHROM", "POS", "REF", "ALT", LIST_SUM(LIST_TRANSFORM(GT, lambda GT: GT IS NOT NULL)) AS n_called FROM statpopgen;
-- 6.
--
-- Collect the necessary statistics for a Hardy-Weinberg Equilibrium test. The
-- actual test involves the Levene-Haldane distribution which is somewhat subtle
-- to implement in SQL.
SELECT "CHROM", "POS", "REF", "ALT",
       LIST_SUM(LIST_TRANSFORM(GT, lambda GT: GT == 0)) as N_HOM_REF,
       LIST_SUM(LIST_TRANSFORM(GT, lambda GT: GT == 1)) as N_HET,
       LIST_SUM(LIST_TRANSFORM(GT, lambda GT: GT == 2)) as N_HOM_ALT
  FROM statpopgen;
-- 7. Read just one variant (this is the sixth one).
SELECT *
  FROM statpopgen
 WHERE "CHROM" == 'chr21'
   AND "POS" == 5030278;
-- 8. Read a 700 base-pair window of variants.
SELECT *
  FROM statpopgen
 WHERE "CHROM" == 'chr21'
   AND "POS" >= 5030300
   AND "POS" <= 5031000;
-- 9.
--
-- Count all the the alternate alleles across all genotypes at this position (nb: a single position,
-- e.g. chr21:123456, may have many "variants", e.g. `A, G`, `A, T`, `A, AGGTG`).
-- NB: The dataset is sorted by chrom, pos, ref, and alt.
  SELECT "CHROM", "POS", SUM(LIST_SUM("GT")) AS N_ALTS
    FROM statpopgen
GROUP BY "CHROM", "POS";
-- 10.
--
-- Collect all the genotypes at variants whose minor allele frequency is < 10%.
SELECT "CHROM", "POS", "REF", "ALT", "GT",
       1.0 - CAST(LIST_SUM(GT) AS DOUBLE) / (2 * LIST_SUM(LIST_TRANSFORM(GT, lambda GT: GT IS NOT NULL))) AS reference_allele_frequency
  FROM statpopgen
 WHERE reference_allele_frequency > 0.9 OR reference_allele_frequency < 0.1;
