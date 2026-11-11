# JSONBench benchmark

[JSONBench](https://github.com/ClickHouse/JSONBench) is ClickHouse's benchmark for analytics
over semi-structured data: Bluesky firehose events, one JSON document per row. This suite
stores each document in a single Variant column, `data`, of the table `bluesky`, and runs
JSONBench's five queries. It is the main workload for the Vortex Variant type.

The harness lives in [`src/jsonbench`](../src/jsonbench).

## Data

`data-gen` downloads the gzipped NDJSON source files and parses them into Parquet files with a
[Parquet Variant](https://github.com/apache/parquet-format/blob/master/VariantEncoding.md)
column. The Vortex files are converted from that Parquet, like every other suite, and hold the
column as a Vortex `Variant` in the Parquet Variant encoding.

Two options select the dataset:

- `--opt scale-factor=N`: the number of source files, each one million events. The default
  is 1, and JSONBench's own sizes are 1, 10, 100, and 1000.
- `--opt shredding=none|partial|query-fields`: which fields are
  [shredded](https://github.com/apache/parquet-format/blob/master/VariantShredding.md) into
  typed columns. Fields that are not shredded stay in the binary Variant value.

| Shredding | Typed fields |
| --- | --- |
| `none` (default) | None |
| `partial` | `kind`, `did`, `time_us` |
| `query-fields` | The above plus `commit.collection`, `commit.operation` |

Every option value has its own data directory, `data/jsonbench/<N>m-<shredding>`. The source
files are shared, in `data/jsonbench/json`.

## Queries

Variant access has no SQL syntax that both engines accept, so each engine has its own file in
[`jsonbench/`](./jsonbench). The queries are numbered from Q0 in file order, and return the
same rows on both engines.

- [`datafusion.sql`](./jsonbench/datafusion.sql) calls `variant_get(data, path, type)`. This
  is the signature of [datafusion-variant](https://github.com/datafusion-contrib/datafusion-variant),
  which has no release for the DataFusion and Arrow versions Vortex builds against. Until it
  does, `datafusion-bench` registers its own function with that signature, on the same
  `parquet-variant-compute` kernel.
- [`duckdb.sql`](./jsonbench/duckdb.sql) uses DuckDB's native `VARIANT` type, with field
  access as `data.commit.collection::VARCHAR`.

## Engines and formats

| Engine | Parquet | Vortex | Vortex Compact | DuckDB |
| --- | --- | --- | --- | --- |
| DataFusion | yes | yes | yes | |
| DuckDB | yes | no | no | yes |

DuckDB cannot read the Vortex files yet: the Vortex DuckDB extension has no export from a
Vortex Variant to a DuckDB `VARIANT`. The `duckdb` format needs a `duckdb` CLI with the
`VARIANT` type (1.5 or later) on `PATH`.

## Performance

DataFusion hoists each `variant_get` call into a projection above the scan, and the Vortex
scan absorbs it: `vortex-datafusion` converts the call to the Vortex `variant_get`
expression, in projections and in filters. For a shredded path the scan then reads the typed
column, and never touches the binary Variant value. Parquet has no such pushdown, so
DataFusion reads the whole Variant column and extracts from it.

An unshredded Variant value is an opaque binary blob that the lightweight encodings cannot
compress. The default Vortex preset therefore compresses the binary storage of a Variant with
zstd, as Vortex Compact does, which keeps the unshredded file level with zstd Parquet.

Query time of Vortex relative to Parquet, on one million rows with DataFusion on four cores
(lower is better):

| Shredding | Q0 | Q1 | Q2 | Q3 | Q4 |
| --- | --- | --- | --- | --- | --- |
| `none` | 1.03 | 1.02 | 1.06 | 0.83 | 0.77 |
| `query-fields` | 0.21 | 0.40 | 0.33 | 0.29 | 0.26 |

The file is 126.5 MB against 128.6 MB of Parquet for `none`, and 135.1 MB against 118.6 MB
for `query-fields`.

On `none` both formats spend their time in the same `parquet-variant-compute` kernel, which
parses every Variant once per extracted path. With the queried fields shredded, DataFusion's
aggregation takes most of the remaining time. What is left in Vortex:

1. **A Variant column is one unit of IO.** A chunk holds the metadata, the binary value, and
   every shredded column in a single segment, so a query that needs one shredded field still
   reads all of it. Writing each shredded field as its own column layout, and resolving
   `variant_get` in the layout reader, would let a shredded query read only the fields it
   names. On local disk this is a small part of the shredded queries; on object storage it
   is the bytes fetched.
2. **Shredded chunks compress chunk by chunk.** The compressor merges the chunks of an
   unshredded Variant, but keeps shredded ones apart, so a writer that feeds it small batches
   of shredded data stores many small Variants. `data-gen` avoids this by reading each
   Parquet row group as one batch.
3. **One pass per path.** Extracting several paths of an unshredded Variant parses each
   value once per path. A kernel that extracts them in one pass would speed up `none` for
   Vortex and Parquet alike.
4. **The V2 scan is behind the V1 scan on `none`.** The V1 scan decodes the storage of an
   unshredded Variant once per chunk when an expression extracts from it more than once. The
   V2 scan applies the same step where it evaluates an expression over a Variant batch, but
   its filter and projection reach the column separately, so it still decodes more than once
   and runs `none` at 0.87 to 1.16 of Parquet. On `query-fields` the two scans are level.

## Running locally

```bash
vx-bench run jsonbench --engine datafusion --format parquet,vortex,vortex-compact
vx-bench run jsonbench --engine datafusion --format parquet,vortex --opt shredding=query-fields
vx-bench run jsonbench --engine duckdb --format parquet,duckdb
```
