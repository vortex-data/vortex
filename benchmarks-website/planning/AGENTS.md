<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# AGENTS.md - benchmarks-website v3

Brief for coding agents working on the v3 rewrite of `bench.vortex.dev`.
Keep this file short. Detail belongs in component plans.

## Status

Alpha is shipped. The v3 server, migrator, and inline-charts UI are
all merged to `ct/benchmarks-v3`. The current focus is **production
readiness**: secrets, CI ingestion wiring, smoke-testing on a real
host, the DNS flip, and v2 cleanup. See [`README.md`](./README.md)
for the live punch list.

The v2 site (top-level files in `benchmarks-website/`: `server.js`,
`src/`, `package.json`, `index.html`, `Dockerfile`,
`docker-compose.yml`, `ec2-init.txt`, etc.) is still in production
on `bench.vortex.dev` and **stays running unchanged** until the DNS
flip. The v3 server lives alongside it as `vortex-bench-server` at
`benchmarks-website/server/`.

## Architecture in 10 bullets

- Single Rust binary: `axum` (HTTP) + `maud` (SSR HTML) + embedded
  `duckdb-rs`. All static assets (`chart.umd.js`, `chart-init.js`,
  `style.css`) are `include_bytes!`'d into the binary. No CDN.
- One DuckDB file on local disk holds five fact tables (compression
  time, query measurement, vector search, RAG, random access) plus
  a `commits` dim table. Schema in
  [`01-schema.md`](./01-schema.md).
- One ingest endpoint: `POST /api/ingest`, gated by a static bearer
  token from the `INGEST_BEARER_TOKEN` env var. Wire shapes in
  [`02-contracts.md`](./02-contracts.md).
- Three HTML routes — `/`, `/chart/{slug}`, `/group/{slug}` — and
  one JSON route, `GET /api/chart/{slug}`, all served from the same
  binary.
- `ChartKey` and `GroupKey` enums round-trip through URLs as
  `<prefix>.<base64url(serde_json(...))>` slugs. No DB lookup
  required to decode a URL.
- Charts render inline on the landing page. Each `<canvas>` is
  paired with a `<script id="chart-data-N">` JSON payload that
  `chart-init.js` hydrates lazily via `IntersectionObserver`.
- Per-chart toolbar with zoom-as-scope: each chart fetches up to
  1000 commits once, then the Show / Y / Mode buttons and slider
  adjust the visible range via `chart.update("none")`. Mouse wheel
  pans history. Slider uses `input` events with a 16ms throttle +
  150ms debounce.
- Group ordering is hard-coded to match v2's
  `origin/ct/vfvb:benchmarks-website/index.html` order. Each group
  is wrapped in a `<details>`; only the first is open by default.
- URL state (`?n=&y=&mode=&hidden=`) is honored only on permalink
  pages (`/chart`, `/group`). Landing page resets to defaults on
  open; users customize per-chart in place.
- `vortex-bench-migrate` reads v2 records, runs each through a
  classifier in `migrate/src/classifier.rs`, and either routes the
  record into one of the five fact tables or marks it `Skip(reason)`
  with a typed reason. The run **fails if more than 5% of records
  come back as `Unknown`** — silent data loss is not allowed.

## Code map

| Path | What lives here |
|---|---|
| `benchmarks-website/server/src/main.rs` | Binary entrypoint. Reads `INGEST_BEARER_TOKEN`, `VORTEX_BENCH_BIND` (default `127.0.0.1:3000`), `VORTEX_BENCH_DB` (default `./bench.duckdb`), `VORTEX_BENCH_LOG`. |
| `benchmarks-website/server/src/api.rs` | `chart_payload(conn, &ChartKey, &CommitWindow)` — the shared implementation behind `/api/chart/{slug}`, the inline `<script>` JSON, and `collect_group_charts`. Known N+1 in `collect_group_charts` — flagged with a TODO. |
| `benchmarks-website/server/src/html.rs` | Three HTML routes and the `<details>`-per-group landing page. `LANDING_DEFAULT_N: u32 = 50`. `UiQuery` parses `?n=&y=&mode=&hidden=` on permalink routes. |
| `benchmarks-website/server/src/slug.rs` | `ChartKey` / `GroupKey` enums and `to_slug` / `from_slug` round-trip. |
| `benchmarks-website/server/static/chart-init.js` | Hydration, `IntersectionObserver`, custom external tooltip with delta rows, inline `afterDatasetsDraw` plugin for the dashed crosshair. |
| `benchmarks-website/server/static/style.css` | `.chart-tooltip-host` is `position: absolute; pointer-events: none;` (do not change — fixes the flicker). `.chart-card` is `position: relative`. |
| `benchmarks-website/server/tests/web_ui.rs` | `insta` snapshot tests, seeded by POSTing to `/api/ingest`. No external fixtures. |
| `benchmarks-website/migrate/src/classifier.rs` | `classify_outcome` routes records into a fact table, `Skip(reason)`, or `Unknown`. >5% Unknown gates the run. |

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

Seed test data via the ingest endpoint (the snapshot tests do this
in-process — see `server/tests/web_ui.rs` for the envelope shapes).

Run snapshot tests:

```bash
cargo test -p vortex-bench-server
INSTA_UPDATE=auto cargo test -p vortex-bench-server   # to update
```

For an end-to-end smoke test against migrated data, point
`VORTEX_BENCH_DB` at the output of `vortex-bench-migrate`.

## Repository conventions

See the root [`CLAUDE.md`](/CLAUDE.md) for Rust style, test layout,
and CI norms. Project-specific:

- The v3 server crate lives at `benchmarks-website/server/` and is
  registered in the root `Cargo.toml` `members` list.
- All commits need a `Signed-off-by:` trailer.
- Run `cargo +nightly fmt --all` and narrow clippy on what you
  changed.
- Public-API changes need `./scripts/public-api.sh`.
- Every new public item needs a doc comment.
- Tests return `VortexResult<()>` and use `?`. No `unwrap`.
- Branch from `ct/benchmarks-v3`, not `develop`. PR back to
  `ct/benchmarks-v3`.
- **Never auto-merge**. Open the PR, post the URL, stop. The user
  reviews and merges.

## Things to avoid

- **Don't widen scope past your task.** If a feature feels missing,
  check [`deferred.md`](./deferred.md) and the "Deferred UI
  follow-ups" section of [`README.md`](./README.md) first — it is
  almost certainly already deferred.
- **Don't write a server-side classifier for live ingest.** The
  emitter is responsible for v3-shape records. The migrator's
  classifier exists only to translate v2 records once.
- **Don't rebuild a global page-level toolbar.** Controls are
  per-chart. This was a real failure mode the first time around —
  the page-level toolbar drove every chart together, which is not
  what users want.
- **Don't bind a slider's reactive logic to `change` events.** Use
  `input` events with a small throttle + debounce, otherwise the
  slider only updates on release and feels broken.
- **Don't refetch on scope change.** Each chart fetches a generous
  window once; scope buttons + slider operate on that buffer via
  `chart.update("none")`.
- **Don't re-introduce `pointer-events: auto` on the tooltip
  host.** The tooltip is positioned at the cursor; making it
  pointer-interactive causes a flicker loop. Keep it
  `pointer-events: none` and offset via
  `transform: translate(12px, 12px)`.
- **Don't drift from contracts.** Wire-shape changes are a
  coordinated PR across emitter, migrator, and server.
- **Don't touch the v2 React/Node app.** It stays in production
  unchanged until the DNS flip. The v2 cleanup is its own PR, post-
  flip.
- **Don't reach for WASM.**

## Working branches

| Branch | Purpose |
|---|---|
| `develop` | Live v2 site. Don't break. |
| `ct/benchmarks-v3` | Integration branch carrying the planning commit + landed component PRs. All v3 branches start here. |
| `claude/benchmarks-v3-<topic>` | Per-task feature branches, branched from `ct/benchmarks-v3` and PR'd back to it. |

## How to update this file

Keep it short. If you've learned something a future agent will need:

- Cross-component contract → [`02-contracts.md`](./02-contracts.md)
- Local detail → your component plan
- Decided → [`decisions.md`](./decisions.md)
- Not designing yet → [`deferred.md`](./deferred.md)
- Cross-cutting agent norm → here
