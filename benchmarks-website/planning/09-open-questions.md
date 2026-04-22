<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 09 - Open questions

Decisions that still need human input (or a follow-up agent session) before
or during implementation, plus a log of resolved ones for context.

## Parking lot - revisit with high priority

### ★ Write buffering / HA gap

The single-EC2 design means: if the server is down when CI tries to POST,
CI retries a few times and then fails. The results aren't *lost* (they're
in the CI artifact), but they don't land in the DB without human
intervention.

Fine in principle, fragile in practice. The concrete mitigation we have in
mind but haven't committed to:

- Add a `--spool s3://vortex-ci-benchmark-results/outbox/<run_id>/` flag to
  `scripts/post-ingest.py`. If the POST fails after all retries, write the
  payload to the outbox prefix.
- A scheduled GitHub Action (hourly cron) drains the outbox by POSTing
  each stored payload, deleting on success.

This effectively gives us at-least-once delivery without bringing back the
whole "per-shard merger" infra that we rejected. It's small - maybe 80
LOC of Python + a yaml file. Worth adopting for launch if the risk
tolerance for "server down = manual replay" is low.

**Decision needed**: adopt the spool-to-S3 fallback at launch, or accept
manual replay as the v3 failure mode?

## Architecture

### Q1. Where does the shared classifier crate live?

The classifier (raw measurement → structured row) is shared between the
Leptos server's `/api/ingest` route and the one-shot historical migrator.
Candidate homes:

- New crate `benchmarks-website/shared/` or `benchmarks-website/classifier/`
  (colocated with the website).
- New crate at workspace root (e.g. `vortex-bench-classifier`).

The migrator is its own binary that might not pull in axum/leptos, so it
can't just be a module inside the server crate. Lean toward: one small
`benchmarks-website-shared` crate with the raw-measurement types +
`classify()` + hash function, depended on by both
`benchmarks-website-server` and `benchmarks-website-migrator`.

### Q2. Leptos vs. fallback stack

Check Leptos's state when implementation starts:

- Is SSR-with-hydration stable?
- Are breaking changes imminent?

Fallback: axum + `askama`/`maud` templates with vanilla JS for the chart
interactivity. DB and schema decisions are framework-agnostic either way.

### Q3. Post-launch auth upgrade path

Launch config is a shared bearer token (see [`07-ingestion.md`](./07-ingestion.md)).
Three viable upgrade paths, pick one post-launch when we want to retire
the shared secret:

- AWS ALB + Cognito + GitHub OIDC (native to our EC2 setup).
- Cloudflare Tunnel + Access (simpler ops, adds Cloudflare).
- Server-side GitHub JWKS validation (no new infra).

Not a launch blocker. Left open.

## Schema / data

### Q4. Rendering model for chart pages

Depends on Q2. If we go with Leptos SSR, do we embed chart data in the
server-rendered HTML (one round trip, larger HTML) or fetch it in a
follow-up request (two round trips, smaller HTML)? Not a schema or
infra question - a frontend detail to settle when we know the framework.

### Q5. Downsampling strategy

v2 does LTTB at 1x/2x/4x/8x and picks a level based on zoom. v3 may not
need it if SSR + raw data is fast enough. Options if we do:

- SQL window function at query time.
- Server-memoized per `(group, chart, range, level)`.
- Materialized views.

Leave this open until we have a v3 site to measure. The DB schema doesn't
need to change either way.

### Q6. `all_runtimes_ns` placement

Today Shape B records carry individual run times as a `LIST<BIGINT>` on
each measurement row. If we ever want variance/error bars in the UI, we
have the data. Current plan: keep in-row. Revisit if storage ever matters
(it won't at our size).

### Q7. Memory benchmark UI

Records are stored; no UI yet. Post-launch task. See
[`08-website.md`](./08-website.md) for the scope. Cheap addition - one
chart per SQL-suite group showing `peak_physical_memory` over time per
engine:format.

## Website UX

### Q8. Per-group vs. global filters

v2 has per-group engine filters. Ship per-group first (parity); consider a
"only show me vortex across every chart" global toggle later.

### Q9. Ad-hoc SQL page

Safe-ish with a read-only handle + timeouts + row limits. Launch-blocker
or v3.1 feature? Lean: v3.1.

### Q10. Commit-diff UI shape

`/commit/:sha` shows one commit's state across every benchmark. What does
"diff vs. parent" look like in the UI? Defer until the page exists and
we can iterate.

## Resolved decisions (log)

### ✓ Ingestion concurrency model

Rejected S3 CAS and the per-shard-merger variant. Settled on: the Leptos
server owns the DB on a local EBS volume; CI POSTs to an authenticated
`/api/ingest`. See [`07-ingestion.md`](./07-ingestion.md).

### ✓ Primary key on `measurements`

Synthetic hash PK (`measurement_id = xxhash64(dimensional tuple)`). NULLs
stay as NULLs in the columns; the ingester canonicalizes NULL into the
hash input. See [`05-schema.md`](./05-schema.md).

### ✓ `filter_sql` in group definitions

Dropped. Group definitions live in typed Rust code, not in data.

### ✓ Extensibility for non-row-table benchmarks

Added `data_descriptor JSON` to `measurements`. Covers vector-search,
future microbenchmarks, anything parameterized. No per-`metric_kind`
schema for the JSON - it's intentionally schemaless.

### ✓ Launch auth model

Shared bearer token validated against an env var. Matches v2's security
complexity.

### ✓ Commit metadata flow

CI includes full commit metadata in every `/api/ingest` payload. The
`commit-metadata` CI job also POSTs commit-only payloads on every push to
`develop` so `commits` is populated for commits with no measurements.
Server never reaches out to the GitHub API.

### ✓ No v2 API backward compatibility

v3 designs fresh `/api/*` route shapes. Nothing external is scripted
against v2's surface.

### ✓ PR benchmark data

Not in v3. `.github/workflows/bench-pr.yml` stays as a PR-local check; its
output does not feed v3.

### ✓ File-size measurements

First-class input, not a sidecar. Migrator reads all historical
`file-sizes-*.json.gz`. Going forward, CI folds file-size records into the
main `/api/ingest` payload. Rendered under the "Compression Size" group.

### ✓ Unclassified records

Go to an `unclassified_records` sidecar table. Never silently dropped. Can
be re-processed with an improved classifier later. See [`05-schema.md`](./05-schema.md).

### ✓ Known engines / formats / datasets bootstrap

Seed SQL file checked into the repo, re-applied on boot via idempotent
upserts. The ingester does *not* auto-insert - unknown engines get
fallback rendering until someone PRs an update to the seed.

### ✓ SQL view sync

`CREATE OR REPLACE VIEW` on server startup. Schema DDL changes are Rust-
owned migrations keyed on `schema_meta.current_version`. No refinery /
sqlx migration framework.

### ✓ Schema version column

Added `schema_meta` table. Server boot-checks it against a constant the
binary compiles in. Refuses to serve if DB is ahead of the binary.

### ✓ Historical data archival

We still migrate `data.json.gz` + `commits.json` + `file-sizes-*.json.gz`
once at cutover. Post-cutover, they stay archived on S3 but are not
load-bearing - rollback is via DuckDB backup snapshots, not re-migration.
Keep S3 bucket versioning enabled as a guard against accidental `rm`.

## Smaller things worth noting but not blocking

- `scripts/compare-benchmark-jsons.py` on `ct/vfvb` might have useful
  testing-time hooks; re-read before writing the migrator's verification
  step.
- `chartjs-plugin-zoom` + `hammerjs` (touch gestures): keep them.
- "Regressions in last N days" page using percentile windows over
  `measurements` is nice but not launch-critical.
