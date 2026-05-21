<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# AGENTS.md - `benchmarks-website/`

Read [`README.md`](README.md) first for the architecture and the v2/v3
side-by-side situation. Then this file. The root [`CLAUDE.md`](../CLAUDE.md)
covers Rust style, test layout, commit conventions.

## Don't touch the v2 site

Until the cutover PR lands, the top-level v2 files
(`server.js`, `src/`, `index.html`, `vite.config.js`, `package.json`,
`package-lock.json`, `public/`, the top-level `Dockerfile`,
`docker-compose.yml`) and the `publish-benchmarks-website.yml` workflow
are production. Don't edit them as part of unrelated work.

The v3 deploy lives entirely under `server/`, `migrate/`, and `ops/`.
The operator runbook is [`ops/README.md`](ops/README.md).

## v3 specifics

- **Wire shapes are a coordinated change.** [`server/src/records.rs`](server/src/records.rs),
  [`vortex-bench/src/v3.rs`](../vortex-bench/src/v3.rs), and (until cutover)
  [`migrate/src/classifier.rs`](migrate/src/classifier.rs) must agree.
  Bumping a shape means changing all three plus the snapshot fixtures in
  one commit. `SCHEMA_VERSION` is the version literal coupled across two
  named sites: [`server/src/schema.rs`](server/src/schema.rs) (source of
  truth) and [`scripts/post-ingest.py`](../scripts/post-ingest.py) (the
  CI ingest wrapper, which hardcodes it as a Python literal). Bump in
  lockstep or every CI ingest run 400s. The server-side validation in
  `records.rs` + `ingest.rs` and the echo in `/health` all consume the
  constant through `crate::schema`.
- **Numeric `?n=` is clamped to 1000; `?n=all` is uncapped.** The HTML
  routes still default to `?n=all` (the no-server-side-cap commitment for
  the default path is honored). The numeric path is bounded by
  `MAX_NUMERIC_COMMIT_WINDOW` in [`server/src/api/window.rs`](server/src/api/window.rs)
  as a DoS-protection floor against `curl ...?n=99999999`. If you need
  full history, use `?n=all`. Do NOT raise the numeric cap or remove it
  without thinking about the DoS surface.
- **`measurement_id` is server-internal.** Never put it on the wire. It is
  a deterministic hash over `commit_sha` plus the dim tuple, computed in
  [`server/src/db.rs`](server/src/db.rs) and reused by the migrator via
  the same crate.
- **Don't write a server-side classifier for live ingest.** The emitter
  produces v3-shape records directly; the migrator's classifier only
  exists to translate v2 records once and goes away after cutover.
- **Don't reach for WASM.** SSR + a thin hydration script in
  [`server/static/chart-init.js`](server/static/chart-init.js) is the
  whole client.
- **v3 ingest is no longer best-effort in CI.** The `Ingest results to v3
  server` step in [`bench.yml`](../.github/workflows/bench.yml),
  [`sql-benchmarks.yml`](../.github/workflows/sql-benchmarks.yml), and
  [`v3-commit-metadata.yml`](../.github/workflows/v3-commit-metadata.yml)
  no longer carries `continue-on-error: true`. A v3-server outage on a
  develop push now fails the bench workflow and triggers the existing
  `incident.io` alert. The gate is `vars.V3_INGEST_URL != ''` so forks
  and unconfigured environments are unaffected.
- **Don't re-introduce a server-side commit cap on `?n=all`.** The HTML
  routes default to `?n=all`; visual downsampling happens client-side via
  LTTB on the visible commit range only. Numeric `?n=` is clamped per the
  bullet above; the unbounded path is `?n=all`.
- **Don't refetch on every scope change.** Once a chart's payload is in
  memory, pan/zoom/slider/range-strip all rebuild in place via the
  in-memory LTTB pass on the cached payload. The single exception is the
  latest-100 to full-history zoom-out path: charts initially hydrate from
  the materialized latest-100 group shard artifact (served from
  `/api/artifacts/{generation}/groups/{slug}/shards/{i}`); when the user
  zooms past that window for the first time, `chart-init.js` lazy-fetches
  `?n=all` once and replaces the latest-100 payload in place.

## Footguns we have already hit

- **Reverse predecessor walk in the tooltip.** `payload.commits[]` is
  sorted oldest-first by SQL - `commits[0]` is the oldest, `commits[N-1]`
  is the newest. For per-row delta the predecessor of `commits[idx]` is
  at `idx - 1`. We caught a regression where a "fix" flipped this to
  `idx + 1`; the original walk-backward direction is right.
- **`pointer-events: auto` on the tooltip host.** The tooltip is
  positioned at the cursor; making it pointer-interactive causes a
  flicker loop. Keep it `pointer-events: none` and offset via
  `transform: translate(12px, 12px)`.
- **`change` events on the slider.** Use `input` events with a small
  throttle; `change` only fires on release and feels broken.

## Local dev

```bash
# Public-only run (read API + ingest only, admin routes 404):
INGEST_BEARER_TOKEN=dev cargo run -p vortex-bench-server

# With admin endpoints mounted on a separate loopback listener:
INGEST_BEARER_TOKEN=dev ADMIN_BEARER_TOKEN=dev \
  cargo run -p vortex-bench-server

cargo nextest run -p vortex-bench-server -p vortex-bench-migrate
INSTA_UPDATE=auto cargo nextest run -p vortex-bench-server   # update snapshots
```

For the full env-var contract (admin bind, snapshot dir, extension dir,
logging spec, PaaS `PORT` fallback) see [`ops/config/vortex-bench.env.example`](ops/config/vortex-bench.env.example)
and the lib-level `//!` doc on [`server/src/main.rs`](server/src/main.rs).

For the migrator end-to-end against the real S3 dump:

```bash
cargo run -p vortex-bench-migrate -- run --output ./bench.duckdb
VORTEX_BENCH_DB=./bench.duckdb INGEST_BEARER_TOKEN=dev \
  cargo run -p vortex-bench-server
```
