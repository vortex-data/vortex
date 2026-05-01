<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# AGENTS.md - benchmarks-website v3

Brief for coding agents working on the v3 rewrite of `bench.vortex.dev`. Keep this file short.
Detail belongs in component plans.

## Status

Alpha is shipped. The v3 server, migrator, and inline-charts UI are all merged to
`ct/benchmarks-v3`. The current focus is **production readiness**: secrets, CI ingestion wiring,
smoke-testing on a real host, the DNS flip, and v2 cleanup. See [`README.md`](./README.md) for the
live punch list.

The v2 site (top-level files in `benchmarks-website/`: `server.js`, `src/`, `package.json`,
`index.html`, `Dockerfile`, `docker-compose.yml`, `ec2-init.txt`, etc.) is still in production on
`bench.vortex.dev` and **stays running unchanged** until the DNS flip. The v3 server lives alongside
it as `vortex-bench-server` at `benchmarks-website/server/`.

## Architecture in 10 bullets

- Single Rust binary: `axum` (HTTP) + `maud` (SSR HTML) + embedded `duckdb-rs`. All static assets
  (`chart.umd.js`, `chart-init.js`, `style.css`) are `include_bytes!`'d into the binary. No CDN.
  A `tower-http` `CompressionLayer` wraps every response (gzip/brotli).
- One DuckDB file on local disk holds five fact tables (compression time, query measurement, vector
  search, RAG, random access) plus a `commits` dim table. Schema in
  [`01-schema.md`](./01-schema.md).
- One ingest endpoint: `POST /api/ingest`, gated by a static bearer token from the
  `INGEST_BEARER_TOKEN` env var. Wire shapes in [`02-contracts.md`](./02-contracts.md).
- Three HTML routes — `/`, `/chart/{slug}`, `/group/{slug}` — and four JSON routes —
  `GET /api/groups`, `GET /api/chart/{slug}`, `GET /api/group/{slug}`, `GET /health` — all served
  from the same binary.
- `ChartKey` and `GroupKey` enums round-trip through URLs as `<prefix>.<base64url(serde_json(...))>`
  slugs. No DB lookup required to decode a URL.
- Charts render inline on the landing page. Each `<canvas>` is paired with a
  `<script id="chart-data-N">` JSON payload that `chart-init.js` hydrates lazily via
  `IntersectionObserver`.
- Per-chart toolbar with zoom-as-scope. Each chart fetches its full raw history once
  (`?n=all`); visual downsampling is **client-side LTTB** in `chart-init.js`
  (`MAX_VISIBLE_POINTS = 500`, applied only to the currently visible commit range — zoomed-in
  views render raw). Drag-pan, drag-rectangle-zoom, wheel-pan, the toolbar slider, and a
  horizontal range-scrollbar strip below each chart all drive the same `rebuildVisibleAndUpdate`
  so LTTB and the strip stay in lockstep. A "downsampled · K / N" badge surfaces when LTTB is
  active.
- Group ordering is hard-coded to match v2's `origin/ct/vfvb:benchmarks-website/index.html` order.
  Every group is wrapped in a `<details>`, all collapsed by default. The first group's chart
  payloads are still inlined (capped at `LANDING_INLINE_N = 100` commits) so opening it skips a
  fetch round-trip; `chart-init.js` lazy-fetches `?n=all` once when the user zooms past the
  inlined window.
- A sticky filter bar at the top of every page exposes engine/format chips that drive series
  visibility across every chart at once. Clicking a data point opens that commit's PR (parsed
  from `(#NNNN)` in the message; falls back to the commit URL). URL params `?engine=&format=&n=`
  survive permalink shares and refreshes; per-chart toolbar state (Y axis, slider) is
  intentionally local-only.
- `vortex-bench-migrate` reads v2 records, runs each through a classifier in
  `migrate/src/classifier.rs`, and either routes the record into one of the five fact tables or
  marks it `Skip(reason)` with a typed reason. The run **fails if more than 5% of records come back
  as `Unknown`** — silent data loss is not allowed.

## Code map

| Path                                             | What lives here                                                                                                                                                                                                            |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `benchmarks-website/server/src/main.rs`          | Binary entrypoint. Reads `INGEST_BEARER_TOKEN`, `VORTEX_BENCH_BIND` (default `127.0.0.1:3000`), `VORTEX_BENCH_DB` (default `./bench.duckdb`), `VORTEX_BENCH_LOG`.                                                          |
| `benchmarks-website/server/src/app.rs`           | `AppState` (DB handle + bearer + path) and the `Router` composition. `CompressionLayer` wraps every response.                                                                                                              |
| `benchmarks-website/server/src/api/`             | Read API. `mod.rs` holds the axum handlers and re-exports the public surface. Submodules: `dto.rs` (wire-shape structs + `GROUP_ORDER`), `window.rs` (`CommitWindow` + `ChartQuery`), `groups.rs` (`collect_groups` discovery passes), `summary.rs` (v2-compatible rollups), `charts.rs` (`chart_payload`, `collect_group_charts`, `SeriesAccumulator`, per-fact-table collectors), `filter.rs` (`collect_filter_universe`). Known N+1 in `collect_group_charts` (charts.rs) — flagged with a TODO. |
| `benchmarks-website/server/src/html/`            | HTML routes. `mod.rs` holds `router()`, `UiQuery`, `FilterState`, the three async page handlers, and `collect_landing_groups`. Submodules: `render.rs` (page chrome, `escape_json_for_script`), `landing.rs` (landing body + chart cards, `LandingGroup`), `chart.rs` (chart and group page bodies), `summary.rs` (group summary cards), `filter.rs` (filter dropdown markup), `toolbar.rs` (per-chart toolbar + range strip), `static_assets.rs` (`include_bytes!`'d JS/CSS/SVG + `STATIC_ASSET_VERSION`). `LANDING_INLINE_N: u32 = 100` caps the first group's inlined chart JSON; HTML routes default to `CommitWindow::All`. |
| `benchmarks-website/server/src/slug.rs`          | `ChartKey` / `GroupKey` enums and `to_slug` / `from_slug` round-trip.                                                                                                                                                      |
| `benchmarks-website/server/static/chart-init.js` | Hydration, `IntersectionObserver`, lazy-fetch on `<details>` toggle, `rebuildVisibleAndUpdate` (client-side LTTB on the visible range, `MAX_VISIBLE_POINTS = 500`), custom external tooltip + delta rows + click-to-PR, range-scrollbar strip, global filter chips, inline crosshair plugin. The canvas state contract (`canvas.__bench_*` fields) and per-card DOM contract (`data-role` selectors) are documented at the top of the file. |
| `benchmarks-website/server/static/style.css`     | `.chart-tooltip-host` is `position: absolute; pointer-events: none;` (do not change — fixes the flicker). `.chart-card` is `position: relative`. `.chart-range-strip*` and `.filter-*` selectors back the range scrollbar and global filter chips. |
| `benchmarks-website/server/tests/`               | `insta` snapshot tests + integration tests, seeded by POSTing to `/api/ingest`. No external fixtures. Tests are split topically (`landing.rs`, `chart_api.rs`, `group_api.rs`, `permalinks.rs`, `static_assets.rs`, `ingest.rs`) sharing fixtures via `common/mod.rs`. |
| `benchmarks-website/migrate/src/migrate/`        | Migration orchestrator (`mod.rs` — `MigrationSummary`, `run`, `apply_v2_record`, `migrate_data_jsonl`, `migrate_file_sizes`, `flush_all`, `open_target_db`) plus per-fact-table accumulators (`accum.rs` — `QueryAccum`, `CompressionTimeAccum`, `RandomAccessAccum`, `CompressionSizeAccum`, `build_*_batch`). |
| `benchmarks-website/migrate/src/classifier.rs`   | `classify_outcome` routes records into a fact table, `Skip(reason)`, or `Unknown`. >5% Unknown gates the run.                                                                                                              |
| `benchmarks-website/migrate/src/verify.rs`       | Structural diff between a migrated DuckDB and v2's live `/api/metadata`. Exits non-zero if any v2 group is missing in v3 — gates a CI step.                                                                                |

## Local dev / smoke test

Build narrow:

```bash
cargo build -p vortex-bench-server
```

Run:

```bash
INGEST_BEARER_TOKEN="dev" cargo run -p vortex-bench-server
# server logs: "bench server listening addr=127.0.0.1:3000 db=bench.duckdb"
```

Seed test data via the ingest endpoint (the snapshot tests do this in-process — see
`server/tests/web_ui.rs` for the envelope shapes).

Run snapshot tests:

```bash
cargo test -p vortex-bench-server
INSTA_UPDATE=auto cargo test -p vortex-bench-server   # to update
```

For an end-to-end smoke test against migrated data, point `VORTEX_BENCH_DB` at the output of
`vortex-bench-migrate`.

## Repository conventions

See the root [`CLAUDE.md`](/CLAUDE.md) for Rust style, test layout, and CI norms. Project-specific:

- The v3 server crate lives at `benchmarks-website/server/` and is registered in the root
  `Cargo.toml` `members` list.
- All commits need a `Signed-off-by:` trailer.
- Run `cargo +nightly fmt --all` and narrow clippy on what you changed.
- Public-API changes need `./scripts/public-api.sh`.
- Every new public item needs a doc comment.
- Tests return `VortexResult<()>` and use `?`. No `unwrap`.
- Branch from `ct/benchmarks-v3`, not `develop`. PR back to `ct/benchmarks-v3`.
- **Never auto-merge**. Open the PR, post the URL, stop. The user reviews and merges.

## Things to avoid

- **Don't widen scope past your task.** If a feature feels missing, check
  [`deferred.md`](./deferred.md) and the "Deferred UI follow-ups" section of
  [`README.md`](./README.md) first — it is almost certainly already deferred.
- **Don't write a server-side classifier for live ingest.** The emitter is responsible for v3-shape
  records. The migrator's classifier exists only to translate v2 records once.
- **Don't rebuild a global page-level toolbar with chart-state controls.** Per-chart controls
  (slider, Y-axis, scope) stay per-chart. The sticky filter bar at the top of every page is the
  exception — it drives series *visibility* across every chart at once, which is what users want
  for the engine/format dimension. Don't extend it with per-chart settings.
- **Don't bind a slider's reactive logic to `change` events.** Use `input` events with a small
  throttle + debounce, otherwise the slider only updates on release and feels broken.
- **Don't refetch every time the scope changes.** The chart fetches its full history once; scope
  buttons, slider, drag-pan, wheel-pan, and the range strip all rebuild via the in-memory LTTB
  pass on the cached payload. The one exception is the inline-payload zoom-out path: when the user
  zooms past the first group's inlined `LANDING_INLINE_N` window for the first time,
  `chart-init.js` lazy-fetches `?n=all` once and replaces the payload.
- **Don't re-introduce a server-side commit cap.** `?n=all` is the default for HTML routes and the
  upper bound is unbounded everywhere. Visual downsampling lives client-side in `chart-init.js`,
  not on the wire.
- **Don't reverse the predecessor walk in the tooltip.** The chart payload's `commits[]` is sorted
  oldest-first by SQL — `commits[0]` is the oldest commit, `commits[N-1]` is the newest. For
  per-row delta the chronological predecessor of `commits[idx]` lives at `idx - 1`. We caught a
  regression where a "fix" flipped this to `idx + 1`; the original walk-backward direction was
  right.
- **Don't re-introduce `pointer-events: auto` on the tooltip host.** The tooltip is positioned at
  the cursor; making it pointer-interactive causes a flicker loop. Keep it `pointer-events: none`
  and offset via `transform: translate(12px, 12px)`.
- **Don't drift from contracts.** Wire-shape changes are a coordinated PR across emitter, migrator,
  and server.
- **Don't touch the v2 React/Node app.** It stays in production unchanged until the DNS flip. The v2
  cleanup is its own PR, post- flip.
- **Don't reach for WASM.**

## Working branches

| Branch                         | Purpose                                                                                             |
| ------------------------------ | --------------------------------------------------------------------------------------------------- |
| `develop`                      | Live v2 site. Don't break.                                                                          |
| `ct/benchmarks-v3`             | Integration branch carrying the planning commit + landed component PRs. All v3 branches start here. |
| `claude/benchmarks-v3-<topic>` | Per-task feature branches, branched from `ct/benchmarks-v3` and PR'd back to it.                    |

## How to update this file

Keep it short. If you've learned something a future agent will need:

- Cross-component contract → [`02-contracts.md`](./02-contracts.md)
- Local detail → your component plan
- Decided → [`decisions.md`](./decisions.md)
- Not designing yet → [`deferred.md`](./deferred.md)
- Cross-cutting agent norm → here
