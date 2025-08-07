SELECT COUNT(*) FROM statpopgen;
SELECT array_length("GT", 1) FROM statpopgen LIMIT 1;
SELECT "GT" FROM statpopgen;
SELECT "CHROM", "POS", "REF", "ALT", 1.0 - CAST(LIST_SUM(GT) AS DOUBLE) / (2 * LIST_SUM(LIST_TRANSFORM(GT, lambda GT: GT IS NOT NULL))) AS reference_allele_frequency FROM statpopgen;
SELECT "CHROM", "POS", "REF", "ALT", CAST(LIST_SUM(GT) AS DOUBLE) / (2 * LIST_SUM(LIST_TRANSFORM(GT, lambda GT: GT IS NOT NULL))) AS alternate_allele_frequency FROM statpopgen;
SELECT "CHROM", "POS", "REF", "ALT", LIST_SUM(LIST_TRANSFORM(GT, lambda GT: GT IS NOT NULL)) AS n_called FROM statpopgen;
SELECT "CHROM", "POS", "REF", "ALT", "GT",
       1.0 - CAST(LIST_SUM(GT) AS DOUBLE) / (2 * LIST_SUM(LIST_TRANSFORM(GT, lambda GT: GT IS NOT NULL))) AS reference_allele_frequency
  FROM statpopgen
 WHERE reference_allele_frequency > 0.9 OR reference_allele_frequency < 0.1;
