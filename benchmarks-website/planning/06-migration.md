<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 06 - Historical data migration

The hard requirement: **no benchmark data is lost in the switch from v2 to v3**.
This doc describes how we get from today's
`s3://vortex-ci-benchmark-results/{data.json.gz, commits.json, file-sizes-*.json.gz}`
blobs into v3's DuckDB, and - just as importantly - how we keep iterating on
that migration during development without pain.

## First-class requirement: a self-contained auto-migrator

One binary. You give it S3 credentials (read-only is fine). It produces a
fully populated v3 DuckDB with every historical benchmark record present and
correctly classified.

This is what enables the "develop in parallel, cut over once it works"
workflow: every time we fix a classifier bug or change the schema, we re-run
the migrator and diff the output against v2's `/api/metadata`. The loop must
be cheap - download, re-ingest, verify should take minutes.

Requirements:

- **Idempotent.** Re-running against the same S3 state produces the same
  DuckDB file, byte-for-byte (or at least row-for-row - `ingested_at`
  timestamps will differ). The deterministic `measurement_id` hash from
  [`05-schema.md`](./05-schema.md) is what makes this work.
- **Self-contained.** No other service needs to be running. The binary can
  write directly to a DuckDB file on local disk. (When a server is running,
  it can also POST in batches to `/api/ingest`; same code path as CI uses.)
- **Dev-friendly.** Runs from a laptop with AWS SSO credentials. Small CLI
  surface: input S3 paths, output DB path, and flags for dry-run / verify.
- **Touches nothing in production.** Default output is a local file or a
  `bench.duckdb.preview` on S3. Never overwrites anything v2 relies on.

## Sources on S3

| S3 key | Size | Shape | Notes |
|--------|------|-------|-------|
| `data.json.gz` | Growing, single-digit MB gzipped | JSONL, one record per measurement (shapes A-C from [`03-raw-data-schema.md`](./03-raw-data-schema.md)) | Main timing/size/memory blob. |
| `commits.json` | Small | JSONL, one record per commit | |
| `file-sizes-*.json.gz` | Small per-dataset | JSONL, size measurements (Shape D) | **First-class input**, not a sidecar. One file per CI benchmark id, written by sql-benchmarks.yml. Data is "how big is a vortex/parquet/lance file for dataset X on commit Y", which users want to see evolve over time - the website will render it as its own "Compression Size" section (same v2 group name). |

All three sources must be fully ingested by the migrator before the new DB
is considered complete. The verification step compares against v2's
`/api/metadata`, which also reads all three sources today.

## The migrator binary

Suggested shape:

```bash
# Run once, dump to a local file, no S3 side-effects.
cargo run -p benchmarks-website-migrator --release -- \
    --data-url      s3://vortex-ci-benchmark-results/data.json.gz \
    --commits-url   s3://vortex-ci-benchmark-results/commits.json \
    --sizes-glob    's3://vortex-ci-benchmark-results/file-sizes-*.json.gz' \
    --output        ./bench.duckdb

# Same classifier, but instead of writing a DB file, POST to a running
# Leptos server's /api/ingest endpoint in batches. This is what you'd use
# in production cutover.
cargo run -p benchmarks-website-migrator --release -- \
    --data-url ...    \
    --commits-url ... \
    --post-to  https://bench-preview.vortex.dev \
    --token    "$INGEST_TOKEN"

# Verify that the resulting DB matches what v2 shows.
cargo run -p benchmarks-website-migrator --release -- verify \
    --db          ./bench.duckdb \
    --against-v2  https://bench.vortex.dev
```

### Internal steps

1. **Read `commits.json`** into memory (it's small). Parse each line into a
   `CommitInfo`. Emit to the `commits` table.

2. **Stream `data.json.gz`**, gunzip on the fly. For each line:
   - Try-parse as Shape A / B / C in turn.
   - Pass the parsed record through **the classifier** (see below).
   - Compute `measurement_id`.
   - Emit to the `measurements` table (INSERT ON CONFLICT DO UPDATE, so
     re-runs don't double-insert).

3. **Stream each `file-sizes-*.json.gz`** with the same classifier.

4. **Run verification** (see below) against v2's `/api/metadata`.

5. (Optional) Upload the resulting DB to the preview key.

### The classifier

This is the crux of the migration. One Rust function, shared with the
ingest endpoint's request handler:

```rust
/// Classify a raw measurement into dimensional columns for the measurements
/// table. Returns None for records that should be dropped (parquet-unc,
/// throughput records, etc. - see v2's server.js::getGroup for the full list).
pub fn classify(raw: &RawMeasurement) -> Option<ClassifiedMeasurement>;
```

Where `RawMeasurement` is `enum { ShapeA(JsonValue), ShapeB(QueryMeasurementJson),
ShapeC(MemoryMeasurementJson) }` and `ClassifiedMeasurement` matches the
`measurements` table shape from [`05-schema.md`](./05-schema.md).

The logic, per shape:

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
  - Anything with `parquet-unc` → **drop** (v2 drops these).
  - Anything with `" throughput"` → **drop** (v2 drops these).
  - Anything that doesn't match → log and drop. Aggregate a count so we can
    see how many records we're losing and triage them.

Port the v2 regex stack **verbatim** to start. Do not refactor during
migration - we want byte-for-byte equivalence with what v2 shows. Refactor in
a follow-up PR once the new site is live and we can diff output easily.

## Verification

The migrator is correct iff it produces the same user-visible chart data as
v2. Two checks:

1. **Count check**: the number of (group, chart, series, commit) tuples in
   v2's `/api/metadata` output equals the number of distinct
   `(benchmark_group, chart_name, series_name, commit_sha)` tuples derivable
   from the new DuckDB via a `SELECT DISTINCT`.
2. **Value check**: pick a sample of (group, chart, series) from v2 and
   compare per-commit values against the DuckDB view. Difference tolerance:
   0 (we haven't done math, just ingested - values must match exactly).

The `verify` subcommand implements both. Run it after every migrator tweak.
Commit a known-good diff summary alongside the migrator so reviewers can see
"before my change, N records were dropped as uncategorized; after, N - 42
are."

## Cutover plan

1. **Dev loop.** Run the migrator → local file. Spin up the Leptos server
   against the local file. Compare charts manually against live v2. Fix
   classifier bugs. Repeat until `verify` passes cleanly.

2. **Preview deploy.** Stand up a separate hostname
   (`bench-preview.vortex.dev`) running the full v3 stack against a preview
   DB on EBS. Run the migrator once, pointed at that server's `/api/ingest`.
   Leave it sitting for a week so we can compare live v2 vs. live v3-preview
   side by side.

3. **Dual-write window.** Wire `scripts/post-ingest.py` into the CI
   workflows (`bench.yml`, `sql-benchmarks.yml`) *in addition to* the
   existing `cat-s3.sh` calls. New runs land in both places. Preview site
   now stays fresh; production site (still v2) is unchanged.

4. **Cut DNS.** Flip `bench.vortex.dev` to the v3 container.

5. **After one quiet week** with no regressions, remove `cat-s3.sh` from the
   CI workflows. `data.json.gz` stops being written; the frozen file stays
   archived on S3 as historical reference but is no longer authoritative.
   Delete the migrator binary source (the DB is now authoritative).

## What we do NOT delete during migration

- `cat-s3.sh`, `commit-json.sh` stay in the repo during the dual-write
  window. After cutover they can go.
- `data.json.gz`, `commits.json`, `file-sizes-*.json.gz` stay archived on
  S3 post-cutover (cheap, and useful as a historical reference if we ever
  need to re-migrate). They are not load-bearing after cutover - rollback
  relies on **DuckDB backup snapshots** (nightly to S3), not on
  re-migrating from raw JSON. Keep the S3 bucket's versioning enabled so
  an accidental `aws s3 rm` is recoverable.

## Estimated scope

- Migrator binary: ~500 LOC Rust including tests and the classifier port. One
  or two engineer-days.
- Verification subcommand: ~100 LOC. Half a day.
- The classifier port itself is where bugs will hide. Schedule a review pass
  that walks through `v2 server.js::getGroup` line by line against the
  migrator.
