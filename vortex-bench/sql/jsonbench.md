# JSONBench

ClickHouse's [JSONBench](https://github.com/ClickHouse/JSONBench): analytical queries over
semi-structured JSON, run against Bluesky social network events. Each event is a JSON document
whose shape depends on its type, so this is the main semi-structured / Variant workload.

The `bluesky` table has a single `data` column holding the whole event. The five queries in
[`jsonbench.sql`](./jsonbench.sql) (numbered from Q0 in file order) are JSONBench's: they read a
handful of JSON paths, filter on some of them and aggregate. The harness lives in
[`src/jsonbench`](../src/jsonbench).

## Formats

Every format is derived from the same raw JSON lines, one output file per one-million-event
input file:

| Format | `data` column |
|---|---|
| `parquet` | the event as a JSON string |
| `parquet-variant` | the event as a shredded Parquet Variant |
| `vortex` | the event as a shredded Vortex Variant, converted from `parquet-variant` |

Both Variant formats shred the same paths. Data generation infers them from the first 100,000
events of the first input file, independently of the queries: every scalar path present in at least
1% of the events whose values share one type in at least 99% of its occurrences. The inferred
paths are saved to `shredding.json` next to the data.

The Vortex files store the Variant column with the Variant layout: the core storage and each
shredded path are separate columns, so reading a path does not decode the rest of the event.

## Queries per engine

A query reads JSON path `a.b` with `{str:a.b}` (a string) or `{i64:a.b}` (a 64-bit integer). The
harness expands these into each engine's idiom for the format:

| Engine | `parquet` | `parquet-variant`, `vortex` |
|---|---|---|
| DataFusion | `json_get_str(data, 'a', 'b')` ([datafusion-functions-json]) | `variant_get(data, 'a.b', 'Utf8')` |
| DuckDB | `json_extract_string(data, '$.a.b')` | `CAST(data."a"."b" AS VARCHAR)` |

DataFusion pushes `variant_get` into Vortex scans, and DuckDB pushes VARIANT field extraction into
Vortex scans, so both read only the queried paths from the Vortex files. Neither engine pushes them
into its Parquet reader.

DuckDB divides integers into doubles, so Q4's `activity_span_ms` has a fractional part on DuckDB
and is an integer on DataFusion.

[datafusion-functions-json]: https://github.com/datafusion-contrib/datafusion-functions-json

## Engine-free runner

`jsonbench-direct` runs the same queries as hand-written plans without a query engine, reported as
the `vortex` engine. Each format extracts the paths in its own way: the Vortex scan evaluates the
filter and `variant_get` projection, Parquet Variant decodes the Variant column with arrow-rs and
extracts the paths, and Parquet JSON parses each event with serde. All formats share one
aggregation, so the comparison isolates how fast each format serves semi-structured data.

## Running locally

The full dataset is 1,000 files of one million events (about 130 GB of compressed JSON). The
`scale-factor` option picks how many millions of events to use (default 1):

```bash
cargo run --release --bin data-gen -- jsonbench --opt scale-factor=10 \
  --formats parquet,parquet-variant,vortex
cargo run --release --bin datafusion-bench -- jsonbench --opt scale-factor=10 \
  --formats parquet,parquet-variant,vortex
cargo run --release --bin duckdb-bench -- jsonbench --opt scale-factor=10 \
  --formats parquet,parquet-variant,vortex
cargo run --release --bin jsonbench-direct -- --opt scale-factor=10
```

Set `VX_BENCH_PRINT_RESULTS=1` to print each query's result, for comparing formats.
