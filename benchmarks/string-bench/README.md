<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# string-bench

Compares Vortex string encoders on configured real-world Utf8 columns.

## What it measures

Three metrics per (column, encoder), from the default `vortex` suite:

| Metric | Meaning | Table unit | `gh-json` unit |
| --- | --- | --- | --- |
| `size` | Serialized file bytes over canonical uncompressed bytes | % | % |
| `write` | Repartition + zone stats + compress (string scheme and children) + layout + serialize | MB/s | ms |
| `read` | Open + scan, decoding each row split to canonical form | MB/s | ms |

Values are the median of `--iterations` runs. The `gh-json` output reports
durations in milliseconds so that lower is better for every emitted metric; the
human table reports MB/s over canonical uncompressed bytes.

### The canonical baseline

Every metric normalizes against canonical uncompressed bytes: one 16-byte view
per row plus the bytes of the strings too long to inline, so a string of 12
bytes or fewer costs only its view. The codec suite uses the same baseline.

### How faithful the timings are

Write and read run the real Vortex writer and scan on a current-thread runtime
with no worker pool, so both timings are single-threaded CPU costs.

`read` fuses the canonical decode into each row split's scan task — the shape
production uses, where `into_record_batch_stream` fuses the Arrow conversion —
and drops each decoded chunk before the next split runs, so steady-state memory
is one chunk, not the whole column. Splits are awaited one at a time rather than
through the scan's own stream, which spawns `concurrency *
available_parallelism()` of them at once; on one thread that read-ahead buys no
parallelism, it only holds more chunks in memory and ties the result to the
host's core count.

Excluded by design:

- **Physical I/O** — the file lives in a `Vec<u8>` and `open_buffer` slices it,
  so there is no read-driver request coalescing, no segment cache, and none of
  the copying a real file or object-store read does.
- **Parallel scan throughput** — production spreads splits across workers, so
  these per-thread costs do not translate directly into query time.
- **Filters and projections** — the whole column is read with no predicate, so
  `read` is the full-decode cost, not decode-with-mask or compute pushdown.

`size` covers encoded children, metadata, padding, and file markers.

### Codec diagnostic

`--suite codec` runs a separate microbenchmark: one whole-column encoder call,
with no Vortex layout, child compression, serialization, or I/O.

It is not part of the tracked set. Its size metric is also not interchangeable
with the file suite's: it trains one dictionary or symbol table for the entire
column and leaves the encoded array's children uncompressed, whereas Vortex
writes in chunks and compresses those children.

## Inputs and local data

The current input catalog is:

| Output name | Source |
| --- | --- |
| `clickbench/URL/shard-N` | ClickBench `hits_N.parquet`, `URL` column |
| `tpch/l_comment` | TPC-H SF1 `lineitem`, `l_comment` column |

On first use, the selected ClickBench shard is downloaded through
`vortex-bench`'s shared idempotent downloader. TPC-H SF1 is generated
deterministically through the shared TPC-H dataset helper. Both are stored
under `vortex-bench/data/`, which is gitignored, and reused on later runs.
Input preparation is outside benchmark timing.

## Running locally

```bash
# Tracked metrics: size, write, read for every configured column and encoder.
cargo run -p string-bench --profile release_debug --features unstable_encodings

# Focus on selected columns or encoders.
cargo run -p string-bench --profile release_debug --features unstable_encodings -- \
  --columns URL --encoders onpair

# Add the direct codec microbenchmark.
cargo run -p string-bench --profile release_debug --features unstable_encodings -- \
  --suite both

# Emit benchmark-comparator JSONL.
cargo run -p string-bench --profile release_debug --features unstable_encodings -- \
  --display-format gh-json --output-path results.json
```

Run `cargo run -p string-bench --features unstable_encodings -- --help` for all
filters and tuning options.

Before timing, the benchmark checks that each requested encoding was produced
and, unless `--no-verify` is set, compares decoded output with the input.

## CI

The develop benchmark workflow runs the default suite after each merge to
`develop` and publishes the results to the shared benchmark history.

Metric names are `<metric>/<input>/<encoder>`, for example
`read/clickbench/URL/shard-0/onpair-12` and `size/tpch/l_comment/fsst`. The unit
is reported separately in the JSON output and the CI table, which groups rows by
unit — so metric-first names keep the rows you compare adjacent.
