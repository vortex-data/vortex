<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 04 - Target architecture

## One-sentence summary

A **Leptos SSR web service** that reads benchmark history out of a **DuckDB
database hosted on S3**, updated by a **small Rust ingester** that replaces the
current `cat-s3.sh` + JSONL blob flow.

## Boxes and arrows

```text
                 +-------------------------+
   CI runner -->|  vortex-bench emits     |
                |  results.json (JSONL)   |
                +-----------+-------------+
                            |
                            v
                +-----------+-------------+
                |  ingester (new)         |
                |  - parse JSONL          |
                |  - classify into dims   |
                |  - INSERT into DuckDB   |
                +-----------+-------------+
                            |
                            v  (single-writer CAS on S3)
                 +----------+---------+
                 | bench.duckdb on S3 |<----------------+
                 +----------+---------+                 |
                            ^                           |
                            | read-only pull on poll    |
                            |                           |
                 +----------+---------+                 |
                 |  Leptos SSR server |                 |
                 |  (EC2 container)   |                 |
                 +----------+---------+                 |
                            |                           |
                            v                           |
                        browsers                        |
                                                        |
          +---------------------------------------------+
          |  one-shot historical migrator (new)
          |  reads data.json.gz + commits.json once,
          |  writes the initial bench.duckdb
          +---------------------------------------------+
```

## Component breakdown

### 1. Ingester (new, Rust)

- Small binary that takes a path to a `results.json` and a path/URL to a
  DuckDB database, and appends rows.
- Can run either:
  - **Inside the CI job** that just produced `results.json`, writing to a
    shared DuckDB on S3 with ETag CAS (same concurrency shape as `cat-s3.sh`,
    but DuckDB-aware). This is the "pure" option and keeps the per-job feedback
    loop tight.
  - Or **batched by a separate job** that drains a "pending" S3 prefix of
    per-run JSONL files and writes a new DuckDB snapshot on a schedule. This
    avoids concurrent-writer-to-one-DuckDB issues entirely.
- We lean toward the batched option for simplicity (see
  [`07-ingestion.md`](./07-ingestion.md) and
  [`09-open-questions.md`](./09-open-questions.md) for the trade-off).

- Responsibilities:
  - Normalize `name` + `target` + `dataset` + `storage` into structured columns
    (this is where v2's classifier logic lives now).
  - Deduplicate (same `(commit_id, measurement_key)` should not be inserted
    twice).
  - Write commit metadata (only the first time we see a new commit hash).

### 2. DuckDB database (new)

- Format: `bench.duckdb` (single file), hosted on S3.
- Size: projected <<100 MB even after years of history. DuckDB compresses the
  data fine for a site with millions of rows at the outer edge of the projection.
- See [`05-schema.md`](./05-schema.md) for tables/columns.

### 3. Leptos SSR server (new)

- Single Rust binary built with Leptos + axum (or whatever Leptos's preferred
  server is at the time we implement).
- Startup:
  - Fetches `bench.duckdb` from S3 into local disk.
  - Opens it read-only.
  - Starts serving.
- Periodically (every N minutes, where N is tuned per deploy):
  - Polls S3 for a new database version (ETag or LastModified check).
  - If changed, atomically swaps in a new read-only handle.
- Serves:
  - `GET /` - landing / overview with summary cards.
  - `GET /group/:slug` - one benchmark group with its charts.
  - `GET /chart/:slug` - one chart, full screen.
  - `GET /commit/:sha` - per-commit snapshot across all benchmarks.
  - `GET /api/chart/:slug` - JSON data for a chart (used by hydrated client to
    re-fetch when the user zooms/pans).
  - `GET /api/metadata` - for backward compatibility / scripts.

- Charting: keep Chart.js. Leptos renders the page, Chart.js instances hydrate
  on the client using data embedded in the SSR output + fetched on interaction.
  (Revisit if we want to move to a Rust-native chart library later, but
  Chart.js is well-trodden and not worth replacing in this rewrite.)

### 4. One-shot historical migrator (new)

- A separate binary, written alongside the ingester (probably sharing code).
- Reads the current `data.json.gz` + `commits.json` from S3, emits the
  **initial** `bench.duckdb`, uploads it.
- Exactly once, at cutover.
- See [`06-migration.md`](./06-migration.md) for the plan.

## Hosting and deploy

- **Today** v2 runs in a docker-compose on a small EC2 (see `ec2-init.txt`).
- **v3** keeps the same deploy shape: a `Dockerfile`, a `docker-compose.yml`,
  EC2 + Watchtower.
- v3 container is a single Rust binary, not Node+Vite. Smaller image, faster
  cold start.
- No database server - DuckDB is embedded and the file is pulled from S3 on
  boot / refresh.

## Why DuckDB (and not Postgres / SQLite / Parquet / Vortex)

- **Postgres**: requires a separate process, has auth/networking concerns, and
  at our scale (single-digit GB) it's overkill. We'd have to back it up
  somewhere anyway.
- **SQLite**: viable. We'd lose the columnar compression we get from DuckDB,
  and DuckDB's analytical query performance on the SQL we want (group-by,
  geomean over many rows) is noticeably better. SQLite is the fallback if
  DuckDB proves troublesome in the Rust ecosystem.
- **Parquet**: no append, no query engine. We'd be reinventing DuckDB.
- **Vortex**: the failure mode `ct/vfvb` already hit. Vortex is optimized for
  immutable columnar storage; benchmark history is append-only and queryable.
  Eventually Vortex will be a great fit; today it is not.

## Why Leptos SSR (and not Axum + React / Next.js / HTMX)

- The rest of the repo is Rust. Keeping the website in Rust means CI, release,
  and dependency management are uniform.
- SSR by default means no WASM-driven rendering in the critical path. Charts
  can still hydrate interactively.
- Leptos has a reasonable `server fn` story so calls into DuckDB from route
  handlers are ergonomic.
- HTMX would be fine for simpler sites but the chart interactions we need
  (pan/zoom, downsample switch, engine filter) benefit from a proper
  component model.
- Next.js is out of scope (not in the Rust ecosystem).

## Non-goals (of the architecture)

- Not multi-tenant. One database, one project, one deploy.
- Not high-availability. If the single EC2 dies, the benchmark *data* is safe
  on S3; the site is down until the container restarts.
- Not real-time. Benchmarks take minutes; a 1-5 minute polling lag for the
  website to pick up new data is fine.
- Not write-heavy. Reads >> writes by a huge margin. We are not optimizing for
  concurrent writers.
