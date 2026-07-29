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

The codec suite trains one dictionary or symbol table for the entire column.
Vortex writes data in chunks, so its encoders may train independently per
chunk. The two suites therefore answer different questions and their size
results are not directly interchangeable.

## Inputs and local data

The current input catalog is:

| Output name | Source |
| --- | --- |
| `URL/shard-N` | ClickBench `hits_N.parquet`, `URL` column |
| `l_comment` | TPC-H SF1 `lineitem`, `l_comment` column |

On first use, the selected ClickBench shard is downloaded through
`vortex-bench`'s shared idempotent downloader. TPC-H SF1 is generated
deterministically through the shared TPC-H dataset helper. Both are stored
under `vortex-bench/data/`, which is gitignored, and reused on later runs.
Input preparation is outside benchmark timing.

## Running locally

```bash
# Default codec suite, all configured columns and encoders.
cargo run -p string-bench --profile release_debug

# Focus on selected columns or encoders.
cargo run -p string-bench --profile release_debug -- \
  --columns URL --encoders onpair

# Include the in-memory Vortex write/read suite.
cargo run -p string-bench --profile release_debug --features unstable_encodings -- \
  --suite both

# Emit benchmark-comparator JSONL.
cargo run -p string-bench --profile release_debug -- \
  --display-format gh-json --output-path results.json
```

Run `cargo run -p string-bench -- --help` for all filters and tuning options.
The `vortex` and `both` suites require the package's opt-in
`unstable_encodings` feature; the default suite does not enable unstable
features elsewhere in the workspace.

Before timing, the benchmark checks that each requested encoding was produced
and, unless `--no-verify` is set, compares decoded output with the input.

## CI status

`string-bench` is a workspace member, so workspace CI can compile and test it.
It is not currently listed in the benchmark or PR-benchmark matrices and its
results are not uploaded to the benchmark dashboard.
