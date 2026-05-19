<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# bench.vortex.dev

The website behind `bench.vortex.dev`. The directory currently houses **two
implementations side by side**, run together until the v3 cutover lands:

- **v2** (top-level files: `server.js`, `src/`, `index.html`, `vite.config.js`,
  `package.json`, `Dockerfile`, `docker-compose.yml`, `public/`). The Node +
  React stack that has shipped to production for the life of the site. Built
  and published by
  [`.github/workflows/publish-benchmarks-website.yml`](../.github/workflows/publish-benchmarks-website.yml).
- **v3** (`server/` + `migrate/` + `ops/`). A single Rust binary —
  [`vortex-bench-server`](server/) — that owns a DuckDB file on local disk,
  serves the API, and renders the HTML. Compiles all static assets
  (`chart.umd.js`, `chart-init.js`, `style.css`) into the binary so deploys
  are one file plus a database. Built directly on the EC2 host by
  [`ops/deploy.sh`](ops/deploy.sh) — see [`ops/README.md`](ops/README.md).
  [`migrate/`](migrate/) is a one-shot tool that loads v2's S3 dataset into a
  v3 DuckDB; it is throwaway and goes away after cutover.

Live results are produced by
[`.github/workflows/bench.yml`](../.github/workflows/bench.yml) and
[`.github/workflows/sql-benchmarks.yml`](../.github/workflows/sql-benchmarks.yml),
which CI runs after every push to `develop`. Until cutover the same payload is
emitted to both stacks (v2 via the legacy `--gh-json` path appended to a public
S3 bucket; v3 via `--gh-json-v3` POSTed to `/api/ingest`).

## v3 architecture in one paragraph

`axum` (HTTP) + `maud` (compile-time HTML) + embedded `duckdb-rs` over a single
local DB file. Five fact tables (`query_measurements`, `compression_times`,
`compression_sizes`, `random_access_times`, `vector_search_runs`) plus a
`commits` dim table — see [`server/src/schema.rs`](server/src/schema.rs) for
the column contracts. Three HTML routes (`/`, `/chart/{slug}`,
`/group/{slug}`) and four stable JSON routes (`GET /api/groups`,
`GET /api/chart/{slug}`, `GET /api/group/{slug}`, `GET /health`), plus
versioned group shard artifacts and bearer-gated `POST /api/ingest`. The hot
website path serves precomputed, precompressed latest-100 artifacts from an
in-memory read model; pages render chart shells and hydrate groups via shard
artifacts, while full history warms in the background. See
[`server/ARCHITECTURE.md`](server/ARCHITECTURE.md).

For the per-module crate map and the request-flow walkthrough, see the
`//!` doc on [`server/src/lib.rs`](server/src/lib.rs). The producer side of
the ingest contract lives in
[`vortex-bench/src/v3.rs`](../vortex-bench/src/v3.rs); the historical-data
side in [`migrate/src/classifier.rs`](migrate/src/classifier.rs).

## Local dev

```bash
# v3 server (DuckDB lives at ./bench.duckdb by default).
INGEST_BEARER_TOKEN=dev cargo run -p vortex-bench-server
# server logs: "bench server listening addr=127.0.0.1:3000 db=bench.duckdb"

# v3 historical migrator (writes a fully populated DuckDB the server can open).
cargo run -p vortex-bench-migrate -- run --output ./bench.duckdb
```

Ingest fixture data via the snapshot tests' envelopes (see
[`server/tests/common/mod.rs`](server/tests/common/mod.rs)) or by hand-rolling
a JSONL file and POSTing through `scripts/post-ingest.py`.

```bash
cargo nextest run -p vortex-bench-server -p vortex-bench-migrate
INSTA_UPDATE=auto cargo nextest run -p vortex-bench-server   # update snapshots
```

For the v2 stack:

```bash
cd benchmarks-website
npm install
npm run dev
```

## Deployment

v3 runs as a systemd service on a single EC2 host. The full operator
runbook (first-time install, day-to-day, failure modes) is in
[`ops/README.md`](ops/README.md). Summary:

- A `vortex-bench-deploy.timer` polls `origin/develop` every 60s. If commits
  in the range touch `benchmarks-website/server/`, `benchmarks-website/migrate/`,
  `Cargo.toml`, or `Cargo.lock`, it builds and atomically swaps the binary,
  then verifies `/health`. Otherwise it fast-forwards the working tree and
  exits silently.
- A `vortex-bench-backup.timer` fires hourly: it asks the server to write a
  per-table Vortex snapshot (`schema.sql` plus one `<table>.vortex` file per
  table) via the bearer-gated `/api/admin/snapshot` endpoint, `tar czf`s the
  snapshot directory into `<UTC ts>.tar.gz`, uploads it to
  `s3://vortex-benchmark-results-database/v3-backups/`, and deletes the local
  copies.
- For ad-hoc reads against the live DB, `ops/inspect.sh` calls a
  bearer-gated `/api/admin/sql` endpoint — no server stop required.

The v3 server is throwaway-friendly: every request runs against the local
DuckDB file, and a fresh boot reapplies the schema DDL idempotently. The
migrator deletes the target file (and its `.wal`) before populating it, so
re-running `vortex-bench-migrate run --output ...` is safe.

## Cutover plan (in flight)

The work to flip `bench.vortex.dev` from v2 to v3 is tracked outside this
repo. The relevant code-side bits:

- v3 runs alongside v2 on the same EC2 host today and is fed by CI's
  dual-write `--gh-json-v3` path.
- v2 keeps shipping unchanged until DNS flips. **Do not touch the top-level
  v2 files unless you are doing the cleanup PR opened post-flip.**
- The v2 cleanup PR removes everything top-level under `benchmarks-website/`
  that belongs to v2 (`server.js`, `src/`, `index.html`, `vite.config.js`,
  `package.json`, `package-lock.json`, `public/`, the top-level `Dockerfile`,
  `docker-compose.yml`, and the `publish-benchmarks-website.yml` workflow).
  The v3 tree under `server/`, `migrate/`, and `ops/` is untouched.
