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
comment, on DataFusion and DuckDB over Parquet, Vortex, and vortex-compact files.

The queries use DuckDB's `lambda x: ...` list-lambda syntax. DataFusion parses it under its
DuckDB SQL dialect, which the benchmark enables through `engine_init_sql`. Column names are
quoted because DataFusion lowercases unquoted identifiers.

## Running locally

```bash
vx-bench run statpopgen --engine datafusion,duckdb --format parquet,vortex
```

## Compression and decompression

The CPU compression suite also uses this VCF-derived dataset, with a smaller prefix of
10,000 variants. It retains the full cohort and all typed fields, including the nested
per-sample genotype fields; only the number of variant rows is reduced. This is separate
from the 100,000-variant SQL benchmark above.

```bash
cargo build -p compress-bench --profile release_debug --features lance
target/release_debug/compress-bench --datasets '^statpopgen$' \
  --formats parquet,vortex,vortex-compact,lance --iterations 5
```

The suite measures encoding time, decoding time, and compressed size. Source VCF download
and conversion to staging Parquet happen before the timed measurements.
