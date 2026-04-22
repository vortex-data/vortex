<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 00 - Context

## What this project is

We want to rebuild the Vortex benchmarks website (https://bench.vortex.dev/)
from scratch, one more time. The deliverable is an **axum HTTP server with
compile-time HTML templates** that owns a **DuckDB database** on a local EBS
volume, plus an authenticated `/api/ingest` HTTP endpoint that replaces
today's append-to-gzipped-JSONL dance.

(Earlier drafts of these docs proposed Leptos SSR; we switched to plain
axum + templates for simplicity - see [`04-architecture.md`](./04-architecture.md).)

The project is gated by three hard requirements, in priority order:

1. **Zero data loss.** Every benchmark measurement that exists in
   `s3://vortex-ci-benchmark-results/data.json.gz` today must be present in the
   new DuckDB database when we cut over.
2. **Faster first contentful paint** than the current v2 site.
3. **Easier to extend.** Adding a new benchmark or a new engine:format target
   must not require a round-trip through a 300-line `config.js` file with
   hand-maintained regex rules.

Stretch goals (nice to have but do not block cutover):

- Per-PR result pages linkable from GitHub PR comments.
- Ad-hoc SQL query page for power users.
- Mobile-friendly UI.

## A short history of this website

**v1 (current-minus-two, no longer on `develop`).** A static site that loaded
`data.json.gz` directly from S3 into the browser, uncompressed it in JS, and
rendered Chart.js plots. First contentful paint was ~30s; the JS was a tangle of
worker scripts (`data-worker.js`, `chart-manager.js`, `scoring.js`,
`zoom-sync.js`, etc.); adding new benchmark series was fragile.

**vfvb (hackathon, `ct/vfvb`, 2025).** An attempt to fix v1 by storing benchmarks
in a Vortex file on S3 and rendering the site client-side via WASM + Dioxus.
Failed for two reasons:

- WASM-rendered sites are also slow; the bottleneck was never "client-side JS is
  too slow", it was "download+parse 10s of MB of JSON".
- Vortex files are immutable. Benchmark data is append-only. Supporting append
  would have required either an index-of-files layer or an expensive
  rewrite-the-world loop. Neither was a good fit.

The hackathon did leave behind useful scaffolding though - see
[`02-vfvb-salvage.md`](./02-vfvb-salvage.md).

**v2 (current `develop`).** A pragmatic server-side rewrite in Node:
`benchmarks-website/server.js` downloads `data.json.gz` + `commits.json` every
5 minutes, parses them into in-memory maps, pre-downsamples, and serves
`/api/metadata` + `/api/data/:group/:chart`. Frontend is React + Vite + Chart.js.
FCP is reasonable but:

- The heuristics that turn raw benchmark names into groups/charts/series live in
  `benchmarks-website/src/config.js` and `benchmarks-website/server.js` as
  hand-tuned rules (`QUERY_SUITES`, `BESPOKE_CONFIGS`, `CHART_NAME_MAP`,
  `ENGINE_RENAMES`, etc.). They drift from the benchmark emitters over time.
- Data is still a gzipped JSONL blob that has to be fully re-read on every
  refresh. There is no query layer - everything is a full scan.
- The raw JSONL schema is underdefined (see
  [`03-raw-data-schema.md`](./03-raw-data-schema.md)): the `name` field smuggles
  group / dataset / query index / engine / format as slash-delimited strings,
  and different measurement types use different conventions.

## Why a third attempt

The current v2 site is "fine" but the underlying data model is still the v1
data model, and every new benchmark series adds complexity to the viewer rather
than to the data. An axum SSR site on top of a real database lets us:

- Make the schema explicit and enforce it at the ingestion boundary, not in the
  viewer.
- Answer new UI questions with SQL instead of by writing more JS.
- Keep the site fast (SSR, no WASM runtime for core chart pages).
- Stay in the Rust ecosystem (matches the rest of the repo, easier for the team
  to maintain).

## Non-goals

- We are not building a general-purpose performance tracker. The dataset is
  small (tens of MB, single-digit millions of rows).
- We are not going to store raw benchmark traces (pprof, flamegraphs). That
  stays in Polar Signals / CI artifacts.
- We are not replacing the CI runners or `vortex-bench` itself. We change the
  output *sink*, not the measurement logic.
- We are not building a PR-blocking regression detector in this project. That
  can be built later on top of the new DB.
