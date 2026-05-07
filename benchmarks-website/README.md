<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# bench.vortex.dev

The website behind `bench.vortex.dev`. The directory currently houses **two
implementations side by side**, run together until the v3 cutover lands:

- **v2** (top-level files: `server.js`, `src/`, `index.html`, `vite.config.js`,
  `package.json`, `Dockerfile`, `docker-compose.yml`, `ec2-init.txt`,
  `public/`). The Node + React stack that has shipped to production for the
  life of the site. Built and published by
  [`.github/workflows/publish-benchmarks-website.yml`](../.github/workflows/publish-benchmarks-website.yml).
- **v3** (`server/` + `migrate/`). A single Rust binary —
  [`vortex-bench-server`](server/) — that owns a DuckDB file on local disk,
  serves the API, and renders the HTML. Compiles all static assets
  (`chart.umd.js`, `chart-init.js`, `style.css`) into the binary so deploys
  are one file plus a database. Container image at
  `ghcr.io/vortex-data/vortex/vortex-bench-server:latest`.
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
`/group/{slug}`) and four JSON routes (`GET /api/groups`,
`GET /api/chart/{slug}`, `GET /api/group/{slug}`, `GET /health`), plus a
bearer-gated `POST /api/ingest`. Charts render inline on the landing page via
SSR + lazy hydration; visual downsampling (LTTB at most
`MAX_VISIBLE_POINTS = 500`) is client-side in
[`server/static/chart-init.js`](server/static/chart-init.js).

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

`docker-compose.yml` runs both stacks side by side: v2 on `:80` and v3 on
`:3001`. `watchtower` polls GHCR every 60s so a fresh image push lands
automatically. v3 reads `INGEST_BEARER_TOKEN` from
`/etc/vortex-bench/secrets.env`, persists DuckDB to
`/opt/benchmarks-website/data/bench.duckdb`, and binds `0.0.0.0:3000` so the
container's `:3001` host port forwards through.

The v3 server is throwaway-friendly: every request runs against the local
DuckDB file, and a fresh boot reapplies the schema DDL idempotently. The
migrator deletes the target file (and its `.wal`) before populating it, so
re-running `vortex-bench-migrate run --output ...` is safe.

## Cutover plan (in flight)

The work to flip `bench.vortex.dev` from v2 to v3 is tracked outside this
repo. The relevant code-side bits:

- v3 runs alongside v2 on the same EC2 host today (v2 on `:80`, v3 on
  `:3001`) and is fed by CI's dual-write `--gh-json-v3` path.
- v2 keeps shipping unchanged until DNS flips. **Do not touch the top-level
  v2 files unless you are doing the cleanup PR opened post-flip.**
- The v2 cleanup PR removes everything top-level under `benchmarks-website/`
  that belongs to v2 (`server.js`, `src/`, `index.html`, `vite.config.js`,
  `package.json`, `package-lock.json`, `public/`, the top-level `Dockerfile`,
  `docker-compose.yml`, `ec2-init.txt`, and the
  `publish-benchmarks-website.yml` workflow). The v3 tree under `server/` and
  `migrate/` is untouched.
