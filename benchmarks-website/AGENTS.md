<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# AGENTS.md — `benchmarks-website/`

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
  one commit. `SCHEMA_VERSION` has a fourth coupled site —
  [`scripts/post-ingest.py`](../scripts/post-ingest.py), the CI ingest
  wrapper — which hardcodes the version as a Python literal; bump it in
  lockstep or every CI ingest run 400s.
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
- **Don't re-introduce a server-side commit cap.** `?n=all` is the default
  for HTML routes; visual downsampling happens client-side via LTTB on the
  visible commit range only.
- **Don't refetch on every scope change.** Once a chart's payload is in
  memory, pan/zoom/slider/range-strip all rebuild in place via the
  in-memory LTTB pass on the cached payload. The single exception is the
  latest-100 → full-history zoom-out path: charts initially hydrate from a
  `LANDING_INLINE_N`-sized shard payload (the materialized latest-100
  artifact, served from `/api/artifacts/{generation}/groups/{slug}/shards/{i}`);
  when the user zooms past that window for the first time,
  `chart-init.js` lazy-fetches `?n=all` once and replaces the latest-100
  payload in place. `LANDING_INLINE_N` is the legacy name for the
  latest-100 window default; nothing is inlined into the landing HTML
  anymore.

## Footguns we have already hit

- **Reverse predecessor walk in the tooltip.** `payload.commits[]` is
  sorted oldest-first by SQL — `commits[0]` is the oldest, `commits[N-1]`
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
INGEST_BEARER_TOKEN=dev cargo run -p vortex-bench-server
cargo nextest run -p vortex-bench-server -p vortex-bench-migrate
INSTA_UPDATE=auto cargo nextest run -p vortex-bench-server   # update snapshots
```

For the migrator end-to-end against the real S3 dump:

```bash
cargo run -p vortex-bench-migrate -- run --output ./bench.duckdb
VORTEX_BENCH_DB=./bench.duckdb INGEST_BEARER_TOKEN=dev \
  cargo run -p vortex-bench-server
```
