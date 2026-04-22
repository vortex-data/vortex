<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Reference snapshots from `ct/vfvb`

These files are **read-only reference snapshots** extracted from the archived
`ct/vfvb` hackathon branch. They are here so future agents working on the v3
rewrite don't have to fish them out with `git show origin/ct/vfvb:...`.

## Do not treat this as live code

- These `.rs` files **do not compile** against current `develop`. `ct/vfvb`
  branched off a much older `vortex-array` API; struct-array builders,
  `to_canonical`, `ToCanonical`, `FieldNames`, and many other types have since
  moved or changed shape.
- `benchmarks-website/` is not a Cargo workspace member, so these files are not
  picked up by `cargo build`. They are docs in a directory that happens to
  contain `.rs` files for editor syntax highlighting.
- If you port any of this code into v3, **rewrite it against current HEAD.
  Do not copy line-for-line.**

## What's here and what to do with it

### Direct port candidates (after API update)

| File | Origin | Use |
|------|--------|-----|
| [`commit_id.rs`](./commit_id.rs) | `vortex-wasm/src/website/commit_id.rs` | 20-byte SHA-1 newtype with hex serde + passthrough `Hasher`. Essentially complete; update imports only. |
| [`commit_info.rs`](./commit_info.rs) | `vortex-wasm/src/website/commit_info.rs` | `CommitInfo { timestamp, author{name,email}, message, commit_id }`. Keep the plain struct + serde; discard the Vortex array conversions (v3 writes to DuckDB). |
| [`entry.rs`](./entry.rs) | `vortex-wasm/src/website/entry.rs` | `BenchmarkEntry { commit_id, group_name, chart_name, series_name, value }`. The hackathon's narrow intermediate representation. v3 keeps the *idea* (one row per measurement) but with structured dimension columns (see `planning/05-schema.md`), not pre-joined group/chart/series strings. Useful as a sketch of the normalized form; not the final schema. |
| [`migrate_data.rs`](./migrate_data.rs) | `vortex-wasm/src/bin/migrate_data.rs` | JSON → Vortex migrator skeleton. **This is the single most useful file** for v3's historical migrator. Swap the output from a Vortex file to a DuckDB insert loop; port v2's classifier (from `benchmarks-website/server.js::getGroup`) into the per-record transformation step. |
| [`migrate_commits.rs`](./migrate_commits.rs) | `vortex-wasm/src/bin/migrate_commits.rs` | Same pattern as `migrate_data.rs` but for commit metadata. In v3 it becomes `INSERT INTO commits`. |

### Reference-only (read, do not copy)

| File | Origin | Use |
|------|--------|-----|
| [`charts.rs`](./charts.rs) | `vortex-wasm/src/website/charts.rs` | The "align series to sorted commit list" logic: given `Vec<BenchmarkEntry>` + sorted `Vec<CommitInfo>`, produce nested `group → chart → series → Vec<Option<NonZeroU64>>` aligned with commits. In v3 this becomes a SQL query (join measurements against commits ordered by timestamp). Useful as a reference for *what the shape of a chart's aligned series* should look like in JSON responses. |
| [`update_s3.rs`](./update_s3.rs) | `vortex-wasm/src/website/update_s3.rs` | AWS-CLI-driven ETag compare-and-swap: `head-object` for ETag, download, transform, `put-object --if-match`, retry on 412. v3 does not use this exact flow (see `planning/07-ingestion.md` - we prefer per-shard writes + a separate merger), but the error-handling shape here is a reasonable reference if we ever need CAS from Rust. |
| [`vortex-wasm.Cargo.toml`](./vortex-wasm.Cargo.toml) | `vortex-wasm/Cargo.toml` | The hackathon's crate manifest, with native (tokio + fs) and WASM (wasm-bindgen + reqwest) feature gates and two migrator binaries. v3's ingester crate will want a similar `native`-feature split if the migrator and a runtime ingestion service share code, minus all the WASM bits. |
| [`hackathon-plan.md`](./hackathon-plan.md) | root `plan.md` | The original hack-week plan doc. Includes the schema sketch that informed `entry.rs` and the "1 file vs many files" discussion. Worth a read for historical context; do not port forward. |
| [`v2-classifier.js`](./v2-classifier.js) | `benchmarks-website/server.js` lines 56-145 | The v2 Node classifier (`rename`, `getGroup`, `formatQuery`, `normalizeChartName`). **This is the single source of truth the migrator's one-shot classifier must reproduce bug-for-bug.** Port line-by-line into Rust; do not work from the prose description in `../03-raw-data-schema.md`. Deleted from the main repo post-cutover along with the migrator binary. |
| [`v2-config-top.js`](./v2-config-top.js) | `benchmarks-website/src/config.js` lines 1-80 | v2's `QUERY_SUITES`, `FAN_OUT_GROUPS`, and `ENGINE_RENAMES` tables. Used by `getGroup` + needed when porting the group-filter enum (see `../11-implementation-kickoff.md`) and the seed SQL for `known_engines` / `known_formats`. |

### Explicitly not copied

Several files from the hackathon branch were **intentionally left behind**:

- `vortex-wasm/src/website/read_s3.rs` - HTTP + Vortex reader + WASM-friendly
  caching. v3 reads DuckDB, not Vortex, from local disk, not HTTP. The whole
  shape is wrong for v3.
- `vortex-wasm/src/website/mod.rs` - WASM bindings (`load_benchmark_summary`,
  `load_chart_data`). v3 has no WASM in the critical path.
- `vortex-wasm/src/lib.rs` - just WASM init glue.
- `wasm-test/`, `vortex-wasm/wasm-test.{html,js,css}` - demo pages.
- The `benchmarks-website/` directory on `ct/vfvb` - that's the v1 vanilla-JS
  site, superseded by v2 on `develop`.

If a future agent decides they need any of these, they're recoverable from the
`ct/vfvb` branch directly: `git show origin/ct/vfvb:<path>`.

## Cross-references

- See [`../02-vfvb-salvage.md`](../02-vfvb-salvage.md) for the fuller narrative
  on what's useful and why.
- See [`../06-migration.md`](../06-migration.md) for how the migrator-style
  files here get turned into the v3 historical migrator.
- See [`../05-schema.md`](../05-schema.md) for how `entry.rs`'s
  `BenchmarkEntry` differs from v3's `measurements` table.
