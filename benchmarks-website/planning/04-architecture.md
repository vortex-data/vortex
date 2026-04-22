<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 04 - Target architecture

## One-sentence summary

A **Leptos SSR web service** that owns a local **DuckDB database** on an EBS
volume, reads it to render pages, and accepts authenticated HTTP POSTs from
CI to ingest new benchmark results.

## Boxes and arrows

```text
                 +-------------------------+
   CI runner -->|  vortex-bench emits     |
                |  results.json (JSONL)   |
                +-----------+-------------+
                            |
                            | POST /api/ingest  (OIDC-auth)
                            v
            +---------------+----------------+
            |  Leptos SSR server             |
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

### 1. Leptos SSR server (new, Rust)

- Single binary, single process. Built with Leptos (SSR mode) + axum.
- Holds a read-write DuckDB handle. All request handlers share it.
- Serves SSR HTML for browser routes, JSON for `/api/*` routes.
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

Not its own long-running service - it's **a route on the Leptos server**:
`POST /api/ingest`. The classifier (the "parse raw vortex-bench JSON →
structured row" logic ported from v2's `server.js::getGroup`) is a library
in its own crate that the server depends on.

The same library backs the one-shot historical migrator (component 4).

### 4. One-shot historical migrator (new)

- Reads `data.json.gz` + `commits.json` from S3.
- For each record: runs the classifier, produces a structured row.
- Either (a) POSTs in batches to a running server's `/api/ingest`, or
  (b) writes directly into a local DuckDB file when running without a
  server (for dev / testing).
- Idempotent (deterministic `measurement_id` hash). Re-running it is free.
- One binary, one job. Delete or archive it post-cutover.

## What runs where

| What | Where |
|------|-------|
| Leptos server | EC2 instance in a Docker container (same host as v2 today) |
| DuckDB file | EBS volume mounted into the container |
| Backups | S3 (`vortex-ci-benchmark-results/backups/`) |
| Historical `data.json.gz` | S3, unchanged, archived for reference |
| Ingest POSTs | HTTPS to the public EC2 / CloudFront hostname |

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

- **Today**: v2 runs in docker-compose on an EC2 (see `ec2-init.txt`).
- **v3**: same shape. Dockerfile + docker-compose.yml + Watchtower on EC2.
  - Attach an EBS volume to the instance. Mount it into the container.
  - Add an S3 write permission (backup) + CI POST access (via CloudFront /
    ALB with auth).
- The container is a single Rust binary instead of Node+Vite. Smaller image,
  faster cold start.

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

## Why Leptos SSR (and not Axum + templates / HTMX / Next.js)

- The rest of the repo is Rust.
- SSR by default means no WASM in the critical path. Charts can still hydrate
  interactively.
- Leptos's `server fn` story makes DuckDB-from-handler ergonomic.
- HTMX + askama/maud is a reasonable fallback if Leptos is in flux at
  implementation time; see [`09-open-questions.md`](./09-open-questions.md).
- Next.js is out of scope (not in the Rust ecosystem).

## Non-goals (of the architecture)

- Not multi-tenant. One database, one project, one deploy.
- Not high-availability. If the EC2 dies, the site is down until the
  container restarts. Data survives on EBS; worst case we restore from a
  backup snapshot.
- Not real-time. A few seconds of delay between "CI POSTs" and "chart shows
  new point" is fine (the Leptos route handler sees the write immediately
  because it's the same process, but any caching on the read path introduces
  a few seconds' lag).
- Not write-heavy. Reads >> writes. We are not optimizing for concurrent
  writers; there is one writer class (CI) funneling through one HTTP
  endpoint.
