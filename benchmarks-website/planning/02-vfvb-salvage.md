<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 02 - What to salvage from `ct/vfvb`

The `ct/vfvb` branch is a hackathon project that tried to replace the v1
benchmarks website with client-side WASM + a Vortex file on S3. The runtime
design is a dead end (see [`00-context.md`](./00-context.md)) but parts of the
code are directly useful as scaffolding for v3's ingestion and migration
tooling.

The files listed here should all be understood as **inspiration and starting
points**, not final code. Expect to rewrite them against the current `develop`
HEAD; `ct/vfvb` branched off a very old `vortex-array` API.

## Useful

### `vortex-wasm/src/website/commit_id.rs`

- 20-byte SHA-1 `CommitId` newtype with hex serde and a passthrough-hasher.
- Straight-up useful. The passthrough hasher in particular is the right move
  for a type that's already a cryptographic hash.
- In v3, `CommitId` probably just lives in the ingestion crate (not in
  `vortex-wasm` which we will likely delete after migration).

### `vortex-wasm/src/website/commit_info.rs`

- Struct: `CommitInfo { timestamp: i64, author: Author { name, email }, message,
  commit_id: CommitId }`.
- Conversion to/from a Vortex struct array.
- Shape is roughly right for a `commits` table in DuckDB. Drop the Vortex
  conversions; keep the Rust struct + serde for the ingester.

### `vortex-wasm/src/website/entry.rs`

- Struct: `BenchmarkEntry { commit_id, group_name, chart_name, series_name,
  value }`.
- This is the **de-facto schema the `ct/vfvb` PR settled on** after reading all
  the data on S3 and deciding what the minimum set of fields needed to be. It
  matches what the current v2 `server.js` reconstructs client-side.
- **Important**: this schema is too narrow for v3. See
  [`05-schema.md`](./05-schema.md) - we want structured columns for
  `dataset, scale_factor, storage, query_idx, engine, format, metric_kind`
  instead of one pre-joined `group_name` / `chart_name` / `series_name` triple.
  But the triple is exactly what the **existing v2 site needs to display**, so
  it's a useful intermediate representation.

### `vortex-wasm/src/bin/migrate_data.rs` and `migrate_commits.rs`

These are the single most valuable pieces of the branch for v3.

- `migrate_data.rs` reads an existing JSON file into `Vec<BenchmarkEntry>`, then
  builds a Vortex file.
- `migrate_commits.rs` same for `CommitInfo`.
- We **keep the "read JSON, normalize, write to structured store" pattern**,
  but swap the destination (Vortex file → DuckDB) and add the classifier logic
  from v2's `server.js::getGroup` + `formatQuery` as the normalization step.
- These are one-shot historical migration binaries. After cutover they can be
  deleted.

### `vortex-wasm/src/website/update_s3.rs`

- AWS-CLI-driven optimistic-concurrency-control wrapper: read ETag, download,
  transform, put-object with `--if-match`, retry on 412.
- Same pattern as `scripts/cat-s3.sh` but in Rust.
- **Probably not reused directly.** See [`07-ingestion.md`](./07-ingestion.md):
  for DuckDB we likely do the updates on one writer (a CI step or a small
  ingester service) with a single-writer lock, not with S3 ETag CAS across
  concurrent CI jobs. But the error-handling shape here is a good reference.

### `vortex-wasm/src/website/charts.rs`

- Processes a `Vec<BenchmarkEntry>` into nested `Benchmarks` / `BenchmarkGroup`
  / `ChartData { aligned_series: HashMap<String, Vec<Option<NonZeroU64>>> }`.
- Aligns each series to the sorted list of commits, filling gaps with `None`.
- Has a `BenchmarkSummary` extractor that returns metadata without values (for
  fast initial page load).
- **Pattern is right, implementation will change.** In v3 this transformation
  happens in SQL (joins against the commits table), not in Rust maps. But the
  shape of the output (per-chart aligned series) is what the Leptos handlers
  will return.

### `plan.md` (root of `ct/vfvb`)

- The goals list, the schema sketch with `CommitId` / `NameId` / `BenchmarkEntry`,
  and the "one file vs many files" discussion are good prior art even though we
  are moving to DuckDB.
- Worth reading once; not worth keeping.

## Not useful

### `vortex-wasm/src/website/read_s3.rs`

- Fetches the Vortex files via HTTP, opens them with the Vortex reader, and
  caches the parsed result in a `OnceLock`.
- Tied to WASM + `reqwest` client-side fetches + Vortex's `OpenOptions` API.
- v3 reads DuckDB, not Vortex. Rebuild from scratch in Leptos.

### `vortex-wasm/src/website/mod.rs` WASM bindings

- `load_benchmark_summary()` / `load_chart_data(group, chart)` exported to JS
  via wasm-bindgen.
- The whole point of v3 is to move away from WASM-driven client rendering.
  Delete.

### `vortex-wasm/` as a crate

- The crate itself was a hackathon vehicle for "everything that needs to work in
  the browser". v3 has no WASM in the critical path, so this crate does not
  need to exist. The migration binaries can live somewhere simpler (e.g. a new
  `benchmarks-website/migrator/` or under `vortex-bench/src/bin/`).

### `benchmarks-website/` on `ct/vfvb`

- This is the v1 vanilla-JS site. It has already been superseded by the current
  v2 React site on `develop`. Ignore.

### `wasm-test/`, `vortex-wasm/wasm-test.{html,js,css}`, demo pages

- Interactive demos from the hackathon. Delete.

## TL;DR of what to copy forward

Direct-port candidates (after API updates):

- `commit_id.rs` (CommitId newtype + hex serde + passthrough hasher)
- `commit_info.rs` (CommitInfo struct + serde)
- `migrate_data.rs` and `migrate_commits.rs` (as the skeletons of the v3
  historical migration binary - point them at DuckDB instead of Vortex)

Reference-only (read, do not copy):

- `update_s3.rs` for the ETag CAS pattern
- `charts.rs` for the "align series to sorted commit list" shape
- `plan.md` for prior-art thinking

Skip entirely:

- Everything WASM-binding-related
- The `vortex-wasm` crate boundary
- `read_s3.rs` specifically, and the public-bucket HTTP fetch model in general
