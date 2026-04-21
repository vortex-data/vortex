<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 06 - Historical data migration

The hard requirement: **no benchmark data is lost in the switch from v2 to v3**.
This doc describes how we get from today's
`s3://vortex-ci-benchmark-results/{data.json.gz, commits.json, file-sizes-*.json.gz}`
blobs into v3's DuckDB.

## Sources

| S3 key | Size estimate | Shape | Notes |
|--------|---------------|-------|-------|
| `data.json.gz` | Growing, single-digit MB gzipped | JSONL, one record per measurement (shapes A-C from [`03-raw-data-schema.md`](./03-raw-data-schema.md)) | The main blob. |
| `commits.json` | Small | JSONL, one record per commit | |
| `file-sizes-*.json.gz` | Small per-dataset | JSONL, size measurements (Shape D) | Per-benchmark-id, appended by sql-benchmarks.yml. |

## One-shot migrator

A new binary (probably `benchmarks-website/migrator/src/bin/migrate_from_jsonl.rs`
or similar - location TBD).

### Inputs

- Path or `s3://` URL to `data.json.gz`.
- Path or URL to `commits.json`.
- Optional: glob for `file-sizes-*.json.gz`.
- Output path for the new `bench.duckdb`.

### Steps

1. **Read `commits.json`** into memory (it's small). Parse each line into a
   `CommitInfo`. Insert into `commits` table.

2. **Stream `data.json.gz`**, gunzip on the fly. For each line:
   - Attempt to deserialize as each of Shape A/B/C in turn (use serde tagged
     enums or try-parse-each).
   - Pass the parsed record through **the classifier** (see below).
   - Emit a `measurements` row.

3. **Stream each `file-sizes-*.json.gz`** with the same classifier.

4. **Verify** (see "Verification" below) against v2's `/api/metadata` output.

5. Upload resulting `bench.duckdb` to
   `s3://vortex-ci-benchmark-results/bench.duckdb`.

### The classifier

This is the crux of the migration. It's the single place we have to reproduce
the logic that is currently spread across v2's `server.js::getGroup`,
`formatQuery`, `normalizeChartName`, `ENGINE_RENAMES`, and the emit-side
`ToJson` impls in `vortex-bench`.

Give it a signature like:

```rust
/// Classify a raw measurement into dimensional columns for the measurements
/// table. Returns None if the record should be dropped (e.g. parquet-unc,
/// throughput records).
fn classify(raw: &RawMeasurement) -> Option<ClassifiedMeasurement>;
```

Where `RawMeasurement` is `enum { ShapeA(JsonValue), ShapeB(QueryMeasurementJson),
ShapeC(MemoryMeasurementJson) }` and `ClassifiedMeasurement` matches the
`measurements` table shape from [`05-schema.md`](./05-schema.md).

The classifier logic, per shape:

- **Shape B (`QueryMeasurementJson`)**: easy. The `dataset` tag enum already
  has everything. `metric_kind = 'query_time'`, `engine = target.engine`,
  `format = target.format`, `query_idx = parse_suffix(name)`, `storage`,
  `scale_factor` and `dataset_variant` from the tagged dataset.

- **Shape C (`MemoryMeasurementJson`)**: same as B with `metric_kind =
  'query_memory'` and the four memory columns populated.

- **Shape A (`JsonValue`)**: parse `name` using the v2-server logic:
  - `random-access/...` → `metric_kind = 'random_access'`.
  - `compress time/...` / `decompress time/...` →
    `metric_kind = 'compression_time'`.
  - `<format> size/...` / `<dataset>:<format> size/...` →
    `metric_kind = 'compression_size'` (unit='bytes').
  - `vortex:<x> ratio ...` → **drop**. Ratios become SQL views in v3.
  - Anything with `parquet-unc` → **drop** (v2 already drops these).
  - Anything with `" throughput"` → **drop** (v2 already drops).
  - Anything that doesn't match → log and drop, but aggregate a count so we
    can see how many records we're losing and triage them.

Port the existing regex stack **verbatim** to start. Do not refactor during
migration; we want byte-for-byte equivalence with what v2 shows. Refactor in a
follow-up PR once the new site is live and we can diff output easily.

### Verification

The migrator is correct iff it produces the same user-visible chart data as
v2. Two ways to check:

1. **Count check**: the number of (group, chart, series, commit) tuples in
   v2's `/api/metadata` output equals the number of distinct
   `(benchmark_group, chart_name, series_name, commit_sha)` tuples derivable
   from the new DuckDB via a `SELECT DISTINCT`.
2. **Value check**: pick a sample of (group, chart, series) from v2 and
   compare per-commit values to the DuckDB view. Difference tolerance: 0.
   (We haven't done math, just ingested - values must match exactly.)

Write a short comparison script; commit it next to the migrator. It is
throwaway after cutover.

### Idempotency

The migrator should be re-runnable. Running it twice against the same inputs
should produce the same output. The deterministic `measurement_id` scheme (see
[`05-schema.md`](./05-schema.md)) is key here; running again hits the upsert
path and doesn't duplicate rows.

## Cutover plan

1. Run the migrator in dev. Verify against v2's `/api/metadata`.
2. Stand up the v3 site pointing at a parallel bucket key (e.g.
   `s3://vortex-ci-benchmark-results/bench.duckdb.preview`). Compare charts
   side-by-side with the existing site for a few days.
3. Flip the ingester into production: it starts writing both to
   `data.json.gz` (old path) **and** to `bench.duckdb` (new path). This gives
   us rollback.
4. Cut DNS / deploy to point `bench.vortex.dev` at the v3 container.
5. After one quiet week, stop writing `data.json.gz` and archive it.

## What we do NOT delete during migration

- `data.json.gz` stays on S3 forever (it's cheap). If the new schema ever
  turns out to have missed a dimension, we can re-run the migrator with an
  improved classifier.
- `commits.json` stays too.
- `cat-s3.sh`, `commit-json.sh` stay in the repo during the dual-write window.

## Estimated scope

- Migrator: ~500 LOC Rust including tests and the classifier port. One or two
  engineer-days.
- Verification script: ~100 LOC. Half a day.
- The classifier port itself is where bugs will hide; schedule a review pass
  that walks through `v2 server.js::getGroup` line by line against the
  migrator.
