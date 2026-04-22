<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 04 - Target architecture

## One-sentence summary

An **axum HTTP server with compile-time HTML templates** (`maud` or
`askama`) that owns a local **DuckDB database** on an EBS volume, reads it
to render pages, and accepts authenticated HTTP POSTs from CI to ingest
new benchmark results.

## Boxes and arrows

```text
                 +-------------------------+
   CI runner -->|  vortex-bench emits     |
                |  results.json (JSONL)   |
                +-----------+-------------+
                            |
                            | POST /api/ingest  (bearer auth)
                            v
            +---------------+----------------+
            |  axum server             |
            |  (single Rust binary)          |
            |                                |
            |   - /                          |
            |   - /group/:slug               |
            |   - /chart/:slug               |
            |   - /commit/:sha               |
            |   - /api/ingest  (write)       |
            |   - /api/chart/:slug  (read)   |
            |   - /health                    |
            |                                |
            |   DuckDB handle (RW, single    |
            |   process, shared across       |
            |   request handlers)            |
            +---------------+----------------+
                            |
                            v
                   /var/lib/bench.duckdb
                     (EBS volume)
                            |
                            v  (nightly cron)
                   s3://.../backups/bench-<date>.duckdb
                            ^
                            |  (restore path only)
                            |
                 +----------+----------+
                 |  One-shot migrator  |
                 |  reads historical   |
                 |  data.json.gz +     |
                 |  commits.json,      |
                 |  POSTs to /api/ingest
                 |  (or writes DB      |
                 |  directly in dev)   |
                 +---------------------+
```

The **one server** is the DB, the API, and the website. That's the whole
deploy. Keeping it in one process removes an entire class of coordination
problems (no CAS, no snapshot polling, no split-brain).

## Component breakdown

### 1. axum server (new, Rust)

- Single binary, single process. Built with `axum` + a compile-time HTML
  template library (`maud` preferred; `askama` is the other option).
- Holds a read-write DuckDB handle. All request handlers share it.
- Serves server-rendered HTML for browser routes, JSON for `/api/*` routes.
- Chart interactivity is **vanilla JS + Chart.js** reading chart data from
  inline `<script type="application/json">` tags in the rendered HTML. No
  WASM, no reactive-framework hydration.
- Accepts authenticated POSTs to `/api/ingest`; see
  [`07-ingestion.md`](./07-ingestion.md) for details.
- Runs a nightly backup cron inside the container.

### 2. DuckDB database (new)

- Format: `bench.duckdb` (single file), on an EBS volume mounted at
  `/var/lib/` (or similar).
- Size: projected well under 100 MB even after years of history.
- See [`05-schema.md`](./05-schema.md) for tables/columns.
- **Not** on S3 for day-to-day reads. S3 is for nightly backups only.

### 3. Ingester (new, Rust)

Not its own long-running service - it's **a route on the axum server**:
`POST /api/ingest`. **No classifier lives in the server.** `vortex-bench`
is extended to emit v3-shape JSON directly (see
[`10-emitter-changes.md`](./10-emitter-changes.md)); the ingest handler is
a serde-validated passthrough that upserts rows straight into
`measurements`. ~50 lines of route-handler code, no string parsing.

### 4. One-shot historical migrator (new)

- Standalone crate or binary under `benchmarks-website/migrator/`, kept
  on the development branch only. **Never lands on `main` / `develop`.**
- Reads the historical v2-shape `data.json.gz` + `commits.json` +
  `file-sizes-*.json.gz` from S3.
- Carries its own v2→v3 classifier (the "parse `name` back into structured
  dimensions" logic ported from v2's `server.js::getGroup`). This is the
  **only place a classifier exists** in the whole plan, and it's
  single-use.
- Either (a) POSTs in batches to a running server's `/api/ingest`, or
  (b) writes directly into a local DuckDB file when running without a
  server (for dev / testing).
- Idempotent (deterministic `measurement_id` hash). Re-running it is free.
- **Deleted alongside its classifier post-cutover.** See
  [`06-migration.md`](./06-migration.md).

## What runs where

| What | Where |
|------|-------|
| axum server   | EC2 instance in a Docker container (same host as v2 today) |
| DuckDB file | EBS volume mounted into the container |
| Backups | S3 (`vortex-ci-benchmark-results/backups/`) |
| Historical `data.json.gz` | S3, unchanged, archived for reference |
| Ingest POSTs | HTTP to the EC2's public hostname (bearer-token gated). TLS upgrade is a post-launch follow-up. |

## Why "server owns the DB" instead of "DB on S3"

Earlier drafts of this doc proposed writing the DuckDB file to S3 and having
the server poll for changes, with CI writes happening via CAS from each CI
job. That adds three moving parts (ETag CAS in every CI job; a pending-shards
prefix; a merger cron) to solve a concurrency problem we don't have. At <100
writes/day from one writer class (our CI), the server can simply own the
write path. No coordination layer needed.

Tradeoffs vs. "DB on S3":

- **Pro**: dramatically simpler. No CAS, no polling, no merger.
- **Pro**: ingestion is synchronous from CI's perspective - the POST's HTTP
  response tells you if your data landed.
- **Pro**: the server can reject a bad payload with a useful error message
  instead of it silently landing in a shard somewhere.
- **Pro**: schema changes don't require coordinating multiple writers.
- **Con**: the server is now a write path, so it needs backups (EBS is
  durable enough, nightly snapshot to S3 covers the long tail).
- **Con**: the server must be up when CI wants to write. This is fine; CI
  retries on failure and we keep `results.json` as a CI artifact for replay
  if it ever isn't.

## Hosting and deploy

Same shape as v2 on the happy path, with one new piece (EBS) and one new
concern (write-path auth):

- **Compute**: EC2 instance, same class as v2. Docker-compose + Watchtower.
- **Storage**: attach an EBS volume (e.g. 20 GiB gp3) and mount it into the
  container at `/var/lib/bench/`. The DuckDB file lives here.
- **Read path**: public HTTP on port 80, exactly like v2. No TLS to start.
  Anyone can read the site.
- **Write path**: `/api/ingest` is served from the same binary on the same
  port, but validates a **shared bearer token** against an env var on every
  request. The token lives in a GitHub Actions secret (same storage v2 uses
  for AWS role creds today). This matches v2's current security posture -
  v2 keeps writes off the website entirely by routing them through AWS IAM
  on S3; v3 moves them onto the website and compensates by checking a
  bearer token.
- **TLS + OIDC upgrade path**: once v3 is live we can put the container
  behind either AWS ALB (native OIDC) or Cloudflare Tunnel + Access for
  TLS. Neither is a launch requirement. See [`09-open-questions.md`](./09-open-questions.md)
  for the follow-up note.
- **Backups**: a cron inside the container snapshots the DuckDB file to S3
  nightly. See [`07-ingestion.md`](./07-ingestion.md).
- **Ops bits**: existing `benchmarks-website/ec2-init.txt` needs two edits:
  attach/mount the EBS volume, and add the `INGEST_BEARER_TOKEN` env var
  to the compose file.

## Schema version guard

The server checks a `schema_meta.current_version` row on boot and refuses
to serve if the DB is at a version newer than the binary expects. This
catches "someone deployed an old container against a DB that's already been
migrated forward" at startup rather than midway through a query. Cheap
insurance; see [`05-schema.md`](./05-schema.md).

## Why DuckDB (and not Postgres / SQLite / Parquet / Vortex)

- **Postgres**: another server process to run, with its own auth and
  networking. At single-digit-GB dataset size it's overkill.
- **SQLite**: viable. We lose the columnar compression and analytical query
  performance DuckDB gives us. SQLite is the fallback if the DuckDB Rust
  crate proves troublesome.
- **Parquet**: no append, no query engine. We'd be reinventing DuckDB.
- **Vortex**: the failure mode `ct/vfvb` already hit. Vortex is optimized for
  immutable columnar storage; benchmark history is append-only and queryable.
  Not a fit today. Maybe in the future.

## Why axum + templates (and not Leptos / HTMX / Next.js)

- The rest of the repo is Rust.
- A compile-time HTML template library (`maud` or `askama`) + axum gives
  us full server-rendered HTML with zero client-side framework churn, zero
  WASM, and zero hydration complexity. The server emits `<script
  type="application/json" id="chart-data">...</script>` for each chart and
  Chart.js reads from it. That's the whole hydration story.
- Earlier drafts proposed Leptos SSR. Dropped because the reactive-
  component model adds real complexity (signals, hydration, framework
  upgrades) that we don't need for a dashboard with ~10 page shapes. axum
  + templates is simpler, a little faster to render, and a lot easier for
  future maintainers to pick up without learning Leptos-isms.
- HTMX: viable alternative to "vanilla JS reads JSON script tag" for
  interactions. Skip for launch; consider later if we want fancier zoom/
  pan UX.
- Next.js: out of scope (not Rust).

## Non-goals (of the architecture)

- Not multi-tenant. One database, one project, one deploy.
- Not high-availability. If the EC2 dies, the site is down until the
  container restarts. Data survives on EBS; worst case we restore from a
  backup snapshot.
- Not real-time. A few seconds of delay between "CI POSTs" and "chart shows
  new point" is fine (the axum route handler sees the write immediately
  because it's the same process, but any caching on the read path introduces
  a few seconds' lag).
- Not write-heavy. Reads >> writes. We are not optimizing for concurrent
  writers; there is one writer class (CI) funneling through one HTTP
  endpoint.
