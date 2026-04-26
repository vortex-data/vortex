<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Decisions

A log of the decisions actually pinned for the alpha. Phase-2
decisions deliberately stay open until we get there - see
[`deferred.md`](./deferred.md).

## Resolved (alpha)

- **Storage backend**: DuckDB on local disk.
- **Single binary**: server is one Rust process - HTTP API + HTML
  routes + DuckDB owner. No separate ingester service, no S3
  coordination layer for writes, no client-side WASM.
- **Server crate**: `vortex-bench-server` at `benchmarks-website/server/`,
  registered as a workspace member.
- **Server-side classifier**: there isn't one. The emitter writes
  v3-shape records directly.
- **Fact-table layout**: one fact table per measurement family
  (`query_measurements`, `compression_times`, `compression_sizes`,
  `random_access_times`, `vector_search_runs`) plus a `commits` dim
  table. **No single wide fact table** with a discriminator column.
  Rationale: the families have genuinely different dim and value
  shapes; merging them either bloats every row with NULLs or splits
  scan results across rows that have to be re-joined. See
  [`01-schema.md`](./01-schema.md).
- **Wire-format discrimination**: each ingest record carries a
  `kind` field that names its destination table. See
  [`02-contracts.md`](./02-contracts.md).
- **`measurement_id` is server-internal**: each fact table has a
  primary key that is a deterministic hash of its dim tuple, used
  for idempotent upsert. The hash is **not on the wire**; the
  emitter never computes it. Algorithm and encoding are the
  server's call.
- **Compression encode vs decode**: a single `compression_times`
  table with an `op ∈ {encode, decode}` column.
- **Compression sizes vs times**: separate tables. Different value
  type (bytes vs ns) and different cardinality (one-shot vs
  iterated).
- **Storage of ratios**: not stored as rows. Computed at read time
  from `compression_sizes`.
- **Auth at alpha**: shared bearer token in an env var,
  constant-time compared. Upgrade paths are deferred.
- **Initial render**: SSR HTML with chart data embedded inline as
  JSON. Client-side hydration runs Chart.js against that data.
- **API backwards compat**: none. v3 designs fresh JSON shapes.
- **CLI shape for emitters**: a new `--gh-json-v3 <path>` flag
  alongside the existing `-d`/`-o` form. Both coexist during alpha;
  consolidating the CLI is deferred.
- **`--gh-json-v3` on-disk format**: JSONL of bare records, one
  per line. The ingest envelope (`run_meta` + `commit`) is added
  by the post-ingest script, not by the Rust emitter.
- **`storage` values**: `nvme` or `s3`. Legacy `gcs` is removed.
  Only `query_measurements` carries `storage`.
- **`scale_factor`**: column on `query_measurements` only,
  nullable. Populated for TPC-H/TPC-DS/StatPopGen/PolarSignals;
  NULL for ClickBench/Fineweb/GhArchive/Public-BI. Categorical
  variants (ClickBench flavor, Public-BI dataset name) go in a
  separate `dataset_variant` column.
- **No JSON escape hatch**: new benchmark parameters become real
  columns.
- **Commit metadata**: included in every `/api/ingest` payload. The
  server never reaches out to GitHub.
- **All-or-nothing transactions in `/api/ingest`**: yes; the
  reported `inserted`/`updated` counts are aggregated across all
  five tables.
- **Per-iteration runtimes**: stored in-row as a list column.
- **Slugs are opaque**: the web-ui treats slugs returned by
  `/api/groups` as opaque strings and feeds them back unmodified
  into `/api/chart/:slug`. The server picks the slug format.

## In use (locked in by the server PR)

These were "recommended" before the server PR landed; they are now
the actual stack in `benchmarks-website/server/Cargo.toml`. The
web-ui agent inherits them by working in the same crate.

- HTTP framework: `axum`.
- Compile-time HTML templates: `maud`.
- DuckDB driver: `duckdb-rs`, version pinned in the server crate's
  `Cargo.toml`.
- Snapshot tests: `insta` (workspace dep).
- Logging: `tracing` (workspace dep).

## Open

Specific column choices may still tighten as the emitter and server
land - the **shape** (five tables, the listed dimensions per table)
is the resolved decision. Phase-2 work is in
[`deferred.md`](./deferred.md): deploy strategy, schema migration
framework, admin auth, CI integration, downsampling, PR comparison
post-cutover, EBS RPO. None of these block the alpha.
