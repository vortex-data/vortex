<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 02 - Wire contracts (alpha)

The cross-component glue between the emitter, the POST script, and
the server. Wire-format only - implementations are local to each
component.

If two components disagree about a shape, **this file is right**
and both update.

## Records are discriminated by `kind`

Each record on the wire carries a `kind` field that picks one of
the [five fact tables](./01-schema.md#tables). The emitter never
decides "what column" - it decides "what kind", and the rest of the
row is that kind's flat field set.

| `kind` | Destination table |
|---|---|
| `query_measurement` | `query_measurements` |
| `compression_time` | `compression_times` |
| `compression_size` | `compression_sizes` |
| `random_access_time` | `random_access_times` |
| `vector_search_run` | `vector_search_runs` |

**Unknown `kind` values cause a 400.** Unknown fields within a known
`kind` also cause a 400. Version skew should fail loudly.

## Per-kind record shapes

All shared metadata first; per-kind fields after.

### `query_measurement`

| Field | Type | Required? | Notes |
|---|---|---|---|
| `kind` | `"query_measurement"` | yes | discriminator |
| `commit_sha` | string | yes | 40-hex lowercase |
| `dataset` | string | yes | `tpch`, `tpcds`, `clickbench`, ... |
| `dataset_variant` | string | optional | ClickBench flavor, Public-BI name |
| `scale_factor` | string | optional | TPC SF; n_rows for StatPopGen / PolarSignals |
| `query_idx` | integer | yes | 1-based |
| `storage` | enum string | yes | `nvme` or `s3` |
| `engine` | string | yes | `datafusion`, `duckdb`, `vortex`, `arrow` |
| `format` | string | yes | `vortex-file-compressed`, `parquet`, `lance`, ... |
| `value_ns` | integer | yes | median timing, ns |
| `all_runtimes_ns` | array&lt;integer&gt; | yes | per-iteration timings (may be empty) |
| `peak_physical` | integer | optional | bytes |
| `peak_virtual` | integer | optional | bytes |
| `physical_delta` | integer | optional | bytes |
| `virtual_delta` | integer | optional | bytes |
| `env_triple` | string | optional | e.g. `x86_64-linux-gnu` |

The four memory fields are populated together (all four or none).

### `compression_time`

| Field | Type | Required? | Notes |
|---|---|---|---|
| `kind` | `"compression_time"` | yes | |
| `commit_sha` | string | yes | |
| `dataset` | string | yes | |
| `dataset_variant` | string | optional | |
| `format` | string | yes | |
| `op` | enum string | yes | `encode` or `decode` |
| `value_ns` | integer | yes | |
| `all_runtimes_ns` | array&lt;integer&gt; | yes | |
| `env_triple` | string | optional | |

### `compression_size`

| Field | Type | Required? | Notes |
|---|---|---|---|
| `kind` | `"compression_size"` | yes | |
| `commit_sha` | string | yes | |
| `dataset` | string | yes | |
| `dataset_variant` | string | optional | |
| `format` | string | yes | |
| `value_bytes` | integer | yes | |

### `random_access_time`

| Field | Type | Required? | Notes |
|---|---|---|---|
| `kind` | `"random_access_time"` | yes | |
| `commit_sha` | string | yes | |
| `dataset` | string | yes | random-access dataset name (e.g. `chimp`, `taxi`) |
| `format` | string | yes | |
| `value_ns` | integer | yes | |
| `all_runtimes_ns` | array&lt;integer&gt; | yes | |
| `env_triple` | string | optional | |

### `vector_search_run`

| Field | Type | Required? | Notes |
|---|---|---|---|
| `kind` | `"vector_search_run"` | yes | |
| `commit_sha` | string | yes | |
| `dataset` | string | yes | e.g. `cohere-large-10m` |
| `layout` | string | yes | `TrainLayout`, e.g. `partitioned` |
| `flavor` | string | yes | `VectorFlavor`, e.g. `vortex-turboquant` |
| `threshold` | number | yes | cosine threshold |
| `value_ns` | integer | yes | per-scan wall time (median of iterations) |
| `all_runtimes_ns` | array&lt;integer&gt; | yes | |
| `matches` | integer | yes | |
| `rows_scanned` | integer | yes | |
| `bytes_scanned` | integer | yes | |
| `iterations` | integer | yes | |
| `env_triple` | string | optional | |

## Ingest envelope

`/api/ingest` accepts one envelope per POST. The envelope wraps a
heterogeneous batch of records (any mix of `kind`s). Required
top-level fields:

- `run_meta`: object with `benchmark_id` (string), `schema_version`
  (integer; `1` at alpha), `started_at` (RFC 3339 timestamp).
- `commit`: object with the columns of the [`commits`
  table](./01-schema.md#commits-dim), keyed by their column names
  with `commit_sha` renamed to `sha`. The server upserts this row
  before applying records.
- `records`: array of per-`kind` records as defined above.

`vortex-bench --gh-json-v3 <path>` writes JSONL of bare records
only. The envelope (`run_meta` + `commit`) is added by the
post-ingest script before POSTing - this keeps the Rust emitter
dependency-light.

The post-ingest script is responsible for filling the `commit`
fields. CI has the SHA from `${{ github.sha }}`; the rest comes
from `git show` or equivalent. See
[`components/emitter.md`](./components/emitter.md).

## HTTP matrix for `POST /api/ingest`

| Condition | Status |
|---|---|
| Happy path | 200 with `{ "inserted": N, "updated": M }` |
| Malformed JSON | 400 |
| Unknown `kind`, unknown field, or per-record validation failure | 400 with the offending record index |
| Missing/invalid bearer token | 401 |
| Schema version newer than server expects | 409 |
| Other server error | 500 |

All-or-nothing per POST: a single failed record fails the whole
batch. The reported `inserted` and `updated` counts are aggregated
across all five tables.

## Authentication header

```text
Authorization: Bearer <token>
```

Compared with constant-time equality on the server. Token comes from
the `INGEST_BEARER_TOKEN` env var.

## Slug grammar (server ↔ web-ui)

The web-ui receives slugs from `/api/groups` and feeds them back
into `/api/chart/:slug`. Slugs are **opaque strings** as far as the
web-ui is concerned: it never parses or constructs them itself,
only echoes what the API returned. The server is free to choose any
slug format, change it without breaking the web-ui, or make it
debuggable (e.g. `qm-tpch-q01-nvme-sf1`) - the only contract is
"`/api/chart/:slug` accepts any slug `/api/groups` returned."

## Read API (alpha)

Two routes - just enough to render one chart page. Field shapes are
not binding; refine during implementation.

### `GET /api/groups`

A flat list of distinct group keys derivable from the data, with
just enough metadata to link to a chart. The server walks each fact
table to produce the group keys defined in
[`01-schema.md`](./01-schema.md#group--chart--series-fit). Every
chart entry includes a `slug` that round-trips through
`/api/chart/:slug`.

### `GET /api/chart/:slug`

Returns the data for one chart: a `display_name`, a `unit`, an
ordered `commits` list (sha + timestamp + first-line message + url),
and a `series` map keyed by series name where each value is an
array aligned to `commits` (with `null` for missing data points).

Per-commit page, zoom/pan, range queries, and the rest of the read
API are deferred. See [`deferred.md`](./deferred.md).
