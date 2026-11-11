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
| `none` | 0.91 | 1.34 | 1.58 | 1.03 | 0.95 |
| `query-fields` | 0.36 | 0.60 | 0.59 | 0.63 | 0.68 |

What still costs time, largest first:

1. **A Variant column is one unit of IO.** A chunk holds the metadata, the binary value, and
   every shredded column in a single segment, so a query that needs one shredded field still
   reads and deserializes all of it. Writing each shredded field as its own column layout, and
   resolving `variant_get` in the layout reader, would let a shredded query read only the
   fields it names. This is what bounds the `query-fields` numbers.
2. **Every extraction decodes the storage again.** Each `variant_get` over an unshredded
   path decompresses the metadata and the value and converts them to Arrow, once per call:
   Q2 extracts four paths in its projection and two in its filter. Extracting several paths
   in one pass, or decoding the storage once per chunk, would remove most of the gap on
   `none`.
3. **The filter and the projection both extract.** DataFusion keeps the filtered paths in the
   projection after the scan has absorbed the filter, so they are extracted twice.
4. **Rows that a shredded column does not cover.** `variant_get` checks the residual value
   for the rows where the typed column is null. It returns early when none of them holds a
   value, and otherwise extracts from every row; scattering the few extracted rows into the
   typed column would avoid that.

## Running locally

```bash
vx-bench run jsonbench --engine datafusion --format parquet,vortex,vortex-compact
vx-bench run jsonbench --engine datafusion --format parquet,vortex --opt shredding=query-fields
vx-bench run jsonbench --engine duckdb --format parquet,duckdb
```
