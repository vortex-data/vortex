<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# string-bench

Compares Vortex string encoders on configured real-world Utf8 columns.

## What it measures

The benchmark canonicalizes and compacts each input before measuring it. Inputs
must be non-empty, all-valid Utf8 columns.

| Suite | Scope | Reported metrics |
| --- | --- | --- |
| `codec` (default) | One whole-column encoder call; no Vortex layout, child compression, serialization, or I/O | Encoded buffers as a percentage of canonical size; compression MB/s |
| `vortex` | Single-threaded in-memory Vortex write, open, scan, and canonicalization | File size as a percentage of canonical size; write, canonicalize, and staged-read MB/s |

All throughputs are decimal MB/s based on canonical uncompressed bytes and the
median of `--iterations` runs. The Vortex suite excludes physical storage I/O.
Its full file size includes encoded children, metadata, padding, and file
markers.

The `gh-json` output reports median phase durations in milliseconds instead of
throughput, along with the same size percentages. Lower values are better for
all emitted metrics.

The codec suite trains one dictionary or symbol table for the entire column.
Vortex writes data in chunks, so its encoders may train independently per
chunk. The two suites therefore answer different questions and their size
results are not directly interchangeable.

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
# Default codec suite, all configured columns and encoders.
cargo run -p string-bench --profile release_debug --features unstable_encodings

# Focus on selected columns or encoders.
cargo run -p string-bench --profile release_debug --features unstable_encodings -- \
  --columns URL --encoders onpair

# Include the in-memory Vortex write/read suite.
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

The develop benchmark workflow runs both suites after each merge to `develop`
and publishes the results to the shared benchmark history.

Machine-readable metric names use the hierarchy
`<scope>/<operation>/<input>/<encoder>`, for example
`codec/compression/clickbench/URL/shard-0/fsst` and
`file/read/canonicalize/tpch/l_comment/onpair-12`. The unit is reported
separately in the JSON output and CI table.
