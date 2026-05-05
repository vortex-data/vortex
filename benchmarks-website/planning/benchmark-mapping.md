<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Existing benchmarks → fact-table mapping

A cross-reference from today's benchmark code to the v3 fact tables
in [`01-schema.md`](./01-schema.md). Use this when implementing
emitter `to_v3_json` (component plan in
[`components/emitter.md`](./components/emitter.md)) or when sanity-
checking that the schema is expressive enough.

If a benchmark in this repo is not listed here, it is either
deferred to phase 2 or out of scope for the bench website.

## Source measurement type → target table

The canonical mapping. The Rust types live in
`vortex-bench/src/measurements.rs` (and per-benchmark crates).

| Source type | Wire `kind` | Target table | Notes |
|---|---|---|---|
| `QueryMeasurement` (paired with `MemoryMeasurement`) | `query_measurement` | `query_measurements` | The two structs collapse into **one** v3 record. Memory fields are omitted if `--track-memory` was off. |
| `TimingMeasurement` (only the random-access variant uses this today) | `random_access_time` | `random_access_times` | |
| `CompressionTimingMeasurement` | `compression_time` (with `op ∈ {encode, decode}`) | `compression_times` | The `op` is decided by which side of `compress-bench`'s timing loop produced it. |
| `CustomUnitMeasurement` with byte unit (sizes) | `compression_size` | `compression_sizes` | A new `CompressionSizeMeasurement` extraction lives in `vortex-bench/src/compress/mod.rs`; the emitter no longer rides on `CustomUnitMeasurement`. |
| `CustomUnitMeasurement` with `ratio` unit | **dropped** | none | Computed at read time from `compression_sizes`. |
| `ScanTiming` (vector-search) | `vector_search_run` | `vector_search_runs` | Carries timing **plus** the three counters in the same row. |

## Per-binary inventory

Every benchmark binary in this repo, the measurement structs it
produces today, and the v3 tables those measurements land in.

### `benchmarks/datafusion-bench`

Runs the SQL query suites with `engine = datafusion`, parameterized
over a `Format` (parquet, vortex-file-compressed, vortex-compact,
arrow, lance via the lance-bench wrapper).

- Produces `QueryMeasurement` (+ `MemoryMeasurement` when
  `--track-memory`) → **`query_measurements`**.
- One row per `(commit, dataset, dataset_variant, scale_factor,
  query_idx, storage, engine = "datafusion", format)`.

### `benchmarks/duckdb-bench`

Same as `datafusion-bench` but with `engine = duckdb`.

- Produces `QueryMeasurement` (+ `MemoryMeasurement` when tracking)
  → **`query_measurements`**, with `engine = "duckdb"`.

### `benchmarks/lance-bench`

Three things in one crate:

1. **Query runner** (`src/main.rs`): `engine = datafusion`,
   `format = lance` only. Produces `QueryMeasurement` (+
   `MemoryMeasurement`) → **`query_measurements`**.
2. **Compression runner** (`src/compress.rs`): produces
   `CompressionTimingMeasurement` + size `CustomUnitMeasurement` →
   **`compression_times`** (with `op ∈ {encode, decode}`,
   `format = lance`) and **`compression_sizes`**
   (`format = lance`).
3. **Random-access runner** (`src/random_access.rs`): produces
   `TimingMeasurement` → **`random_access_times`** with
   `format = lance`.

### `benchmarks/compress-bench`

The compression suite. Per dataset, runs encode + decode against
each enabled `Format` and records the resulting on-disk size.

- `CompressionTimingMeasurement` for encode → **`compression_times`**
  with `op = "encode"`.
- `CompressionTimingMeasurement` for decode → **`compression_times`**
  with `op = "decode"`.
- Byte-unit `CustomUnitMeasurement` (the size entries) →
  **`compression_sizes`**.
- Ratio-unit `CustomUnitMeasurement` (the `vortex:parquet-zstd
  ratio/...` entries) → **dropped**. The reader recomputes ratios
  from `compression_sizes`.

### `benchmarks/random-access-bench`

The random-access "take" timing suite. Datasets here (chimp, taxi,
etc.) are a different namespace from the SQL query suites.

- `TimingMeasurement` → **`random_access_times`**.
- `format` is one of `vortex-file-compressed`, `vortex-compact`,
  `parquet`, `lance`.

### `benchmarks/vector-search-bench`

Cosine-similarity scan over a vector dataset. Each dataset/layout/
flavor combination produces a single `ScanTiming` per scan
configuration.

- `ScanTiming` → **`vector_search_runs`**.
- `dataset` from `VectorDataset` (e.g. `cohere-large-10m`).
- `layout` from `TrainLayout`.
- `flavor` from `VectorFlavor` (compression flavor; the vector-
  search analogue of `format`).
- `threshold`, `iterations` are real columns.
- `query_seed` is **not** stored - it's a deterministic seed for
  the query sampler and not a measurement dimension.

## Per-suite dim values

For SQL query suites (everything that flows through
`query_measurements`), the dim columns are populated as follows:

| `BenchmarkArg` | `dataset` | `dataset_variant` | `scale_factor` | Notes |
|---|---|---|---|---|
| `TpcH` | `tpch` | NULL | TPC SF as string (`"1"`, `"10"`, `"100"`, `"1000"`) | |
| `TpcDS` | `tpcds` | NULL | TPC SF as string | |
| `ClickBench` | `clickbench` | NULL | NULL | The migrate path does not encode the `partitioned` / `single` flavor in `dataset_variant`, so the live emitter also leaves it `NULL` to keep historical and live rows in one group. The active flavor is fixed per CI matrix entry. |
| `StatPopGen` | `statpopgen` | NULL | NULL | The migrate path (v2 → v3 backfill) does not carry a per-record scale factor for this suite, so the live emitter also leaves it `NULL` to keep historical and live rows in one group. |
| `PolarSignals` | `polarsignals` | NULL | NULL | Same as StatPopGen. |
| `Fineweb` | `fineweb` | NULL | NULL | |
| `GhArchive` | `gharchive` | NULL | NULL | |
| `PublicBi` | `public-bi` | dataset name (e.g. `cms-provider`) | NULL | The Public-BI sub-dataset name lives in `dataset_variant`. |

For non-query suites:

- `compress-bench`: `dataset` is the compression dataset name; if
  the suite later grows variants, `dataset_variant` is available.
- `random-access-bench`: `dataset` is the random-access dataset
  name. No variant column on this table.
- `vector-search-bench`: see the [vector_search_runs
  table](./01-schema.md#vector_search_runs).

## What this implies for the emitter

The mapping above is the contract `vortex-bench --gh-json-v3`
implements. Any v3 record an emitter writes today must land in
exactly one of the five tables; if a future measurement type
doesn't fit, that's the signal to add a sixth table (and a sixth
`kind`) rather than overload one of these.

The **historical migrator** will use the same mapping when it lands
(it's deferred - see [`deferred.md`](./deferred.md#historical-data-migration)).
The v2 classifier on `develop` at `benchmarks-website/server.js`
becomes useful then, because the v2 S3 dump pre-dates the
discriminator and we'll have to recover `kind` from name strings.
For new ingest at alpha, no classifier is needed.
