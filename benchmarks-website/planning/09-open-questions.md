<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 09 - Open questions

Decisions still needing human input, plus a log of resolved ones for
context. Most of this document is log at this point - only two open
questions remain that gate implementation, and both are about
post-launch follow-ups.

## Still open

### Q1. Post-launch auth upgrade path

Launch config is a shared bearer token (see
[`07-ingestion.md`](./07-ingestion.md)). Three viable upgrade paths, pick
one post-launch when we want to retire the shared secret:

- AWS ALB + Cognito + GitHub OIDC (native to our EC2 setup).
- Cloudflare Tunnel + Access (simpler ops, adds Cloudflare).
- Server-side GitHub JWKS validation (no new infra).

Not a launch blocker. Revisit after v3 is stably in production.

### Q2. Downsampling strategy

v2 does LTTB at 1x/2x/4x/8x and picks a level based on zoom. v3 may not
need any of it if axum + raw DuckDB queries are fast enough. If we do:

- SQL window function at query time.
- Server-memoized per `(group, chart, range, level)`.
- Materialized views.

Leave open until we have a v3 site to measure. The DB schema doesn't need
to change either way.

## Resolved decisions (log)

### ✓ Ingestion concurrency model

Rejected S3 CAS and per-shard-merger variants. The axum server owns the
DB on a local EBS volume; CI POSTs to an authenticated `/api/ingest`. See
[`07-ingestion.md`](./07-ingestion.md).

### ✓ Primary key on `measurements`

Synthetic hash PK (`measurement_id = xxhash64(dimensional tuple)`). NULLs
stay as NULLs in the columns; the ingester canonicalizes NULL into the
hash input. See [`05-schema.md`](./05-schema.md).

### ✓ `filter_sql` in group definitions

Dropped. Group definitions live in typed Rust code, not in data.

### ✓ Extensibility for non-row-table benchmarks

`data_descriptor JSON` column on `measurements`. Covers vector-search,
future microbenchmarks, anything parameterized. Intentionally schemaless.

### ✓ Launch auth model

Shared bearer token validated against an env var. Matches v2's security
complexity.

### ✓ Commit metadata flow

CI includes full commit metadata in every `/api/ingest` payload. The
`commit-metadata` CI job also POSTs commit-only payloads on every push to
`develop`. Server never reaches out to the GitHub API.

### ✓ No v2 API backward compatibility

v3 designs fresh `/api/*` route shapes.

### ✓ PR benchmark data

Not in v3. `.github/workflows/bench-pr.yml` stays as-is.

### ✓ File-size measurements

First-class input. Migrator reads historical `file-sizes-*.json.gz`;
emitter rewrite folds future size records into the main POST payload.
Rendered under the "Compression Size" group.

### ✓ Compression-ratio records are not emitted

The cross-format ratio `CustomUnitMeasurement`s that `compress-bench`
generates today (`vortex:parquet size/<name>`, `vortex:lance ratio
compress time/<name>`, etc.) are derivable from the raw
`compression_size` / `compression_encode` / `compression_decode`
records. In v3 they are **DuckDB views**, not stored rows. The emitter
stops producing them. See [`10-emitter-changes.md`](./10-emitter-changes.md)
per-measurement-type mapping.

### ✓ v3 on-wire emitter output format

`-d gh-json-v3` writes JSONL of bare `ClassifiedMeasurement` records,
one per line. The `IngestPayload` envelope (`run_meta`, `commit`,
`records`) is constructed by the CI wrapper (`scripts/post-ingest.py`)
before POSTing, not by the Rust emitter. Keeps the emitter
dependency-light and mirrors the existing `-d gh-json` convention.

### ✓ Single benchmark run during dual-write

During the dual-write window, the benchmark loop runs **once** and the
CLI emits both `results.json` (v2-shape) and `results.v3.json` (v3-shape)
side by side. `-d` becomes repeatable, with a paired `-o` per format.
Post-cutover, `-d gh-json` is removed and emission collapses back to a
single output path.

### ✓ `gcs` as a storage target

Not real. The `Storage` enum is closed to `{Nvme, S3}`. A stale doc
comment in the code mentioning `gcs` should be removed when
`TimingMeasurement::storage` migrates to the enum.

### ✓ vector-search-bench v3 emission

Deferred to post-launch. v3 ships without wiring
`benchmarks/vector-search-bench/` through `/api/ingest`. Forward-compat
hooks are preserved so no schema-version bump is needed when it
graduates: `MetricKind::VectorSearch*` variants are reserved in the
enum, `data_descriptor` is designed for free-form fields, and
`CustomUnitMeasurement` stays in-tree. See
[`10-emitter-changes.md`](./10-emitter-changes.md) §"vector-search-bench:
deferred to post-launch".

### ✓ Emitter CLI shape

Format-named output flags: `--table`, `--gh-json <path>`,
`--gh-json-v3 <path>`. Multiple flags may be combined in one
invocation; the benchmark runs once. Replaces the old
`-d <format> -o <path>` pair. Designed in a vacuum; backwards
compatibility with `-d`/`-o` is deliberately not preserved since
every caller is in-tree and updated in the same pass. See
[`10-emitter-changes.md`](./10-emitter-changes.md) §"CLI surface".

### ✓ Unclassified records

Go to an `unclassified_records` sidecar table. In v3 steady state this
count should always be zero; non-zero means an emitter bug or version
skew.

### ✓ Known engines / formats / datasets bootstrap

Seed SQL file checked into the repo, re-applied on boot via idempotent
upserts. The ingester does **not** auto-insert - unknown engines get
fallback rendering until someone PRs an update to the seed.

### ✓ SQL view sync

`CREATE OR REPLACE VIEW` on server startup. Schema DDL changes are Rust-
owned migrations keyed on `schema_meta.current_version`.

### ✓ Schema version column

Added `schema_meta` table. Server boot-checks it against a constant the
binary compiles in.

### ✓ Historical data archival

We migrate `data.json.gz` + `commits.json` + `file-sizes-*.json.gz` once
at cutover. Post-cutover, they stay archived on S3 but are not load-
bearing - rollback is via DuckDB backup snapshots.

### ✓ Write buffering (launch requirement)

`scripts/post-ingest.py --spool s3://.../outbox/` dumps the payload to S3
on unrecoverable POST failure. A scheduled workflow (`drain-ingest-outbox.yml`,
10-minute cron, concurrency-gated) drains the outbox by re-POSTing. At-
least-once delivery without new infrastructure. See
[`07-ingestion.md`](./07-ingestion.md).

### ✓ Framework choice

axum + compile-time HTML templates (`maud` or `askama`). Rejected Leptos
SSR for simplicity. Chart.js continues to draw the charts, reading from
inline `<script type="application/json">` tags.

### ✓ Rendering model

Embed default (`last=100`) chart data inline in SSR HTML as JSON
`<script>` tags. Lazy fetch via `GET /api/chart/:slug?start=&end=` for
zoom/pan interactions. See [`08-website.md`](./08-website.md).

### ✓ Classifier location (single biggest architectural decision)

**The main repo has no classifier.** `vortex-bench`'s emitters gain a new
`-d gh-json-v3` output format that emits v3-shape structured JSON
directly. `/api/ingest` is a serde-validated passthrough. The only
classifier in the plan lives in the one-shot historical migrator binary,
kept on the development branch only, deleted along with the migrator
post-cutover. See [`10-emitter-changes.md`](./10-emitter-changes.md).

### ✓ `all_runtimes_ns` placement

In-row as `LIST<BIGINT>`. DuckDB's columnar layout makes storage nearly
free; preserving the data enables variance bands / error bars without
another migration.

### ✓ Memory benchmark UI

Post-launch feature. Data is already stored; we just need view layer.
See [`08-website.md`](./08-website.md).

### ✓ Per-group vs. global engine filters

Per-group at launch (parity with v2). Global toggle is a post-launch
addition if useful.

### ✓ Ad-hoc SQL page

v3.1 feature, not launch.

### ✓ Commit-diff UI shape

Defer until `/commit/:sha` page exists and we can iterate.

## Smaller things worth noting but not blocking

- `scripts/compare-benchmark-jsons.py` on `ct/vfvb` might have useful
  testing-time hooks; re-read before writing the migrator's verification
  step.
- `chartjs-plugin-zoom` + `hammerjs` (touch gestures): keep them.
- "Regressions in last N days" page using percentile windows over
  `measurements` is nice but not launch-critical.
