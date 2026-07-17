# Statistical and Population Genetics benchmark

A genomics analytics benchmark using the gnomAD v3.1.2 release of the jointly called One
Thousand Genomes + Human Genome Diversity Project (1kG+HGDP) dataset — a prefix of
chromosome 21. The data source is <https://gnomad.broadinstitute.org/>.

The distinguishing feature is the `GT` genotype column: a large variable-length list of
small nullable integers per row, which stresses list encodings and list-aware compute.
The queries in [`statpopgen.sql`](./statpopgen.sql) cover common genomics patterns:
allele-frequency calculations, Hardy-Weinberg equilibrium statistics, variant filtering by
frequency and by locus interval, and random access to a specific variant.

The harness lives in [`src/statpopgen`](../src/statpopgen).

## CI variant

CI runs this suite from local NVMe as the `Statistical and Population Genetics` PR
comment. Only DuckDB is exercised (over Parquet, Vortex, and vortex-compact files),
because the queries rely on DuckDB list lambdas.

## Running locally

```bash
vx-bench run statpopgen --engine duckdb --format parquet,vortex
```
