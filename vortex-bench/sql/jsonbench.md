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
- `--opt shredding=none|partial|query-fields|inferred`: which fields are
  [shredded](https://github.com/apache/parquet-format/blob/master/VariantShredding.md) into
  typed columns. Fields that are not shredded stay in the binary Variant value.

| Shredding | Typed fields |
| --- | --- |
| `none` | None |
| `partial` | `kind`, `did`, `time_us` |
| `query-fields` | The above plus `commit.collection`, `commit.operation` |
| `inferred` (default) | None in the Parquet file. The Vortex writer shreds what the documents have in common |

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
expression, in projections and in filters. Parquet has no such pushdown, so DataFusion reads
the whole Variant column and extracts from it.

A shredded Variant is written with the Variant layout, which stores the Variant as a struct
of its core storage and its shredded child. The struct is written like any other, so each
shredded field is a column of its own, with its own encoding and statistics. A scan that
extracts a shredded path reads that column and nothing else: Q0 on `query-fields` reads
0.5 MB of the 126 MB file, as dictionary codes over 15 strings. The layout is unstable. This
suite enables it through an edition of its own, and other writers do not use it.

An unshredded Variant value is an opaque binary blob that the lightweight encodings cannot
compress. The default Vortex preset therefore compresses the binary storage of a Variant with
zstd, as Vortex Compact does, which keeps the unshredded file level with zstd Parquet.

With `inferred`, the Parquet file is the unshredded one, and the Vortex writer chooses what
to shred. It samples the first chunk of the column, and shreds every object field, up to four
objects deep, that holds a string, an integer, a double, or a boolean in each sampled row that
has it. A row that does not fit keeps its value in the binary Variant, so the file holds the
same documents. Here that shreds every field the queries read, without being told which.

Query time of Vortex relative to Parquet, on one million rows with DataFusion on four cores
and the V2 scan (lower is better):

| Shredding | Q0 | Q1 | Q2 | Q3 | Q4 | Vortex file | Parquet file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `none` | 0.62 | 0.44 | 0.29 | 0.28 | 0.31 | 126.5 MB | 128.6 MB |
| `query-fields` | 0.07 | 0.25 | 0.14 | 0.08 | 0.10 | 126.3 MB | 118.6 MB |
| `inferred` | 0.04 | 0.11 | 0.04 | 0.03 | 0.03 | 163.2 MB | 128.6 MB |

In absolute terms, the `inferred` Vortex file answers the five queries in 12, 70, 32, 19 and
22 ms, where the `none` Vortex file takes 173, 283, 240, 228 and 226 ms.

The `inferred` file is larger because the default preset has no zstd for the shredded string
columns, which hold content hashes and record keys that the lightweight encodings barely
compress. Vortex Compact writes the same shredding in 133.1 MB, and answers the queries in 14,
70, 37, 33 and 30 ms.

The suite runs on the V2 scan, which `vx-bench` selects for it. `datafusion-bench` defaults
to the V1 scan, so pass `--scan-v1 false` when running it directly.

On `none`, a split decodes the storage of a Variant once, however many filters and
projections extract from it, and a string or an integer under object fields is read from the
encoded bytes in place: the names of the path are resolved against each distinct metadata
once, and every object is then scanned for a field id. Parquet goes through the `variant_get`
kernel of `parquet-variant-compute`, which builds a `Variant` over every row and every object
on the way down. About four fifths of the CPU time Vortex still spends on `none` is zstd
decompression of the values, which is what an unshredded Variant costs to read. Shredding is
the way past it, which is what `inferred` does.

What is left:

1. **Shredding on write is this suite's default, not the writer's.** `inferred` enables it
   through the suite's unstable edition. The schema comes from the first chunk alone, and the
   shredding runs on the writer's thread.
2. **Extractions the in-place reader does not cover.** A path through a list, a value that
   needs a cast, and a target other than a string or a 64-bit integer go through the general
   kernel, at Parquet's speed.
3. **Shredded paths the scan cannot prove complete.** The scan reads a shredded column alone
   only when the writer recorded that no row left a value for it in the core storage. Where
   some row did, as for a field that holds strings in some rows and numbers in others, the
   scan reads the core storage as well.

## Running locally

```bash
vx-bench run jsonbench --engine datafusion --format parquet,vortex,vortex-compact
vx-bench run jsonbench --engine datafusion --format parquet,vortex --opt shredding=query-fields
vx-bench run jsonbench --engine datafusion --format parquet,vortex --opt shredding=none
vx-bench run jsonbench --engine duckdb --format parquet,duckdb
```
