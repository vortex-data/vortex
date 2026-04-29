<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 01 - DuckDB schema (alpha)

The persistent data model. **One `commits` dim table plus five fact
tables, one per measurement family.** No lookup tables, no views, no
migration framework; those are deferred (see
[`deferred.md`](./deferred.md)).

## Design principles

1. **One fact table per (dim shape, value shape).** A row in any
   fact table has every value column populated; NULLs only appear
   in genuinely optional dimensions.
2. **No discriminator columns spanning families.** No `metric_kind`
   enum forcing five shapes into one row.
3. **No JSON escape hatch.** New benchmark parameters become real
   columns. Adding a nullable column is cheap; the readability win
   is worth it.
4. **Hashed primary key per table.** Each fact table has a
   `measurement_id` that is a deterministic 64-bit hash of
   `commit_sha` plus that table's dimensional tuple. Including
   `commit_sha` makes every (commit, dim) pair a distinct row -
   that's what the chart pages render as a time series.
   Server-internal; not on the wire.
5. **`commits` is the only dim table.** Engine, format, dataset,
   etc. stay as inline strings; DuckDB's dictionary encoding makes
   a lookup table pointless.
6. **Ratios are not stored.** Computed at query time from
   `compression_sizes`.

## Why five fact tables, not one

The five families have genuinely different shapes:

| Table | Shape sketch |
|---|---|
| `query_measurements` | dataset + query_idx + engine + format + storage → timing **and** memory |
| `compression_times` | dataset + format + op∈{encode,decode} → timing |
| `compression_sizes` | dataset + format → bytes |
| `random_access_times` | dataset + format → timing (different dataset namespace) |
| `vector_search_runs` | dataset + layout + flavor + threshold → timing + counters |

Forcing them into one table either bloats every row with columns
that are NULL for ~99% of rows (`layout`, `flavor`, `threshold`,
`matches`, `rows_scanned`, `bytes_scanned`) or splits scan results
across multiple rows that have to be re-joined to render one chart.

## Group / chart / series fit

The render-time view used by `/api/groups` and `/api/chart/:slug`
is mechanically derivable per table:

| Table | Group key | Chart key | Series key |
|---|---|---|---|
| `query_measurements` | `(dataset, dataset_variant, scale_factor, storage)` | `(dataset, query_idx)` | `(engine, format)` |
| `compression_times` | constant `"Compression"` | `(dataset, dataset_variant)` | `(format, op)` |
| `compression_sizes` | constant `"Compression Size"` | `(dataset, dataset_variant)` | `format` |
| `random_access_times` | constant `"Random Access"` | `dataset` | `format` |
| `vector_search_runs` | `(dataset, layout)` | `(dataset, layout, threshold)` | `flavor` |

The classifier logic in v2's `v2-classifier.js` mostly disappears -
each table already knows what suite it represents.

## Tables

DDL is the server's call. Below is the column contract: name, type
family, and whether it's NOT NULL. The server agent picks exact
DuckDB types, indexes, and constraint syntax.

### `commits` (dim)

| Column | Type | Required? | Notes |
|---|---|---|---|
| `commit_sha` | string | yes (PK) | 40-hex lowercase |
| `timestamp` | timestamptz | yes | |
| `message` | string | optional | first line only |
| `author_name` | string | optional | |
| `author_email` | string | optional | |
| `committer_name` | string | optional | |
| `committer_email` | string | optional | |
| `tree_sha` | string | yes | |
| `url` | string | yes | |

Populated from the envelope on every `/api/ingest` call.

### `query_measurements`

SQL query suites: TPC-H, TPC-DS, ClickBench, StatPopGen,
PolarSignals, Fineweb, GhArchive, Public-BI. Memory columns are
populated when the run was instrumented for memory; NULL otherwise.
Timing and memory share the row because they're produced together
for the same query execution.

| Column | Type | Required? | Notes |
|---|---|---|---|
| `measurement_id` | int64 | yes (PK) | hash of dim tuple |
| `commit_sha` | string | yes | FK to `commits` |
| `dataset` | string | yes | `tpch`, `tpcds`, `clickbench`, ... |
| `dataset_variant` | string | optional | ClickBench flavor, Public-BI name |
| `scale_factor` | string | optional | TPC SF; n_rows for StatPopGen / PolarSignals |
| `query_idx` | int32 | yes | 1-based |
| `storage` | string | yes | `nvme` or `s3` |
| `engine` | string | yes | `datafusion`, `duckdb`, `vortex`, `arrow` |
| `format` | string | yes | `vortex-file-compressed`, `parquet`, `lance`, ... |
| `value_ns` | int64 | yes | median timing, ns |
| `all_runtimes_ns` | list&lt;int64&gt; | yes | per-iteration timings |
| `peak_physical` | int64 | optional | bytes |
| `peak_virtual` | int64 | optional | bytes |
| `physical_delta` | int64 | optional | bytes |
| `virtual_delta` | int64 | optional | bytes |
| `env_triple` | string | optional | e.g. `x86_64-linux-gnu` |

### `compression_times`

Encode/decode timings from `compress-bench`.

| Column | Type | Required? | Notes |
|---|---|---|---|
| `measurement_id` | int64 | yes (PK) | |
| `commit_sha` | string | yes | FK |
| `dataset` | string | yes | |
| `dataset_variant` | string | optional | |
| `format` | string | yes | |
| `op` | string | yes | `encode` or `decode` |
| `value_ns` | int64 | yes | |
| `all_runtimes_ns` | list&lt;int64&gt; | yes | |
| `env_triple` | string | optional | |

### `compression_sizes`

On-disk sizes from `compress-bench`. One-shot, no per-iteration data.
Compression ratios in v2 (`vortex:parquet-zstd ratio/...`) are a
SELECT over this table joined to itself; they're not stored.

| Column | Type | Required? | Notes |
|---|---|---|---|
| `measurement_id` | int64 | yes (PK) | |
| `commit_sha` | string | yes | FK |
| `dataset` | string | yes | |
| `dataset_variant` | string | optional | |
| `format` | string | yes | |
| `value_bytes` | int64 | yes | |

### `random_access_times`

Take-time timings from `random-access-bench`. Different dataset
namespace from `compression_times` - kept in its own table so
dataset filters never have to disambiguate which suite a row
belongs to.

| Column | Type | Required? | Notes |
|---|---|---|---|
| `measurement_id` | int64 | yes (PK) | |
| `commit_sha` | string | yes | FK |
| `dataset` | string | yes | |
| `format` | string | yes | |
| `value_ns` | int64 | yes | |
| `all_runtimes_ns` | list&lt;int64&gt; | yes | |
| `env_triple` | string | optional | |

### `vector_search_runs`

Cosine-similarity scans from `vector-search-bench`. The only family
that emits a timing **plus side counters** for the same scan;
keeping them in one row avoids a 1:N split that has to be re-joined
on read.

| Column | Type | Required? | Notes |
|---|---|---|---|
| `measurement_id` | int64 | yes (PK) | |
| `commit_sha` | string | yes | FK |
| `dataset` | string | yes | e.g. `cohere-large-10m` |
| `layout` | string | yes | `TrainLayout`, e.g. `partitioned` |
| `flavor` | string | yes | `VectorFlavor`, e.g. `vortex-turboquant` |
| `threshold` | double | yes | cosine threshold |
| `value_ns` | int64 | yes | per-scan wall time |
| `all_runtimes_ns` | list&lt;int64&gt; | yes | |
| `matches` | int64 | yes | |
| `rows_scanned` | int64 | yes | |
| `bytes_scanned` | int64 | yes | |
| `iterations` | int32 | yes | not part of the dim hash |
| `env_triple` | string | optional | |

## `measurement_id` hash

Per-table xxhash64 over `commit_sha` plus that table's dimensional
tuple. Including `commit_sha` makes every (commit, dim) pair a
distinct row, which is what the chart pages render as a time
series. The hash is **server-internal** - the wire never carries
it. The server's INSERT path computes it before each
`INSERT ... ON CONFLICT DO UPDATE`, which gives idempotent upsert
on re-emission of the same (commit, dim) pair. Encoding details
(input order, NULL handling, byte layout) are the server's call,
since the value never crosses a process boundary.

When the historical migrator lands (deferred), it reuses the
server's hash function via a shared crate.

## Storage values

`storage` is `'nvme'` or `'s3'`. Legacy `gcs` is dropped. Only
`query_measurements` carries `storage` - the other families don't
fan out by storage backend.

## Schema changes during alpha

There is no migration framework. If you change the schema:

1. Update this doc.
2. Update the server's DDL.
3. Delete any local `bench.duckdb` and re-run.

A real forward-only migration framework lands post-alpha. See
[`deferred.md`](./deferred.md).

## What's intentionally NOT here (deferred)

- `schema_meta` and migration framework.
- `known_engines` / `known_formats` / `known_datasets` lookup
  tables and seed SQL.
- Views (`v_compression_ratios`, `v_latest_per_group`, etc.).
- Pre-downsampled aliases.
- A `microbench_runs` table - reserved as the next family to add
  when microbench results start landing.
