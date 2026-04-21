<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 01 - Current state (v2)

This is a snapshot of the v2 benchmarks-website as of `develop` at the time of
writing. It exists so that anyone working on v3 can understand what they're
replacing without re-reading all the code.

## Components

### CI-side: data producers

Two workflows push into the benchmarks bucket
`s3://vortex-ci-benchmark-results/`:

- `.github/workflows/bench.yml` - runs on every merge to `develop`. Invokes
  `random-access-bench` and `compress-bench` with `-d gh-json -o results.json`,
  then appends to `s3://vortex-ci-benchmark-results/data.json.gz` via
  `scripts/cat-s3.sh`. Also runs `scripts/commit-json.sh > new-commit.json` and
  appends to `s3://vortex-ci-benchmark-results/commits.json`.
- `.github/workflows/sql-benchmarks.yml` - runs TPC-H, TPC-DS, Clickbench,
  StatPopGen, PolarSignals, Fineweb. Same "append to data.json.gz" concatenation
  pattern.

`scripts/cat-s3.sh` is an optimistic-concurrency-control shell script: it uses
`aws s3api head-object`'s ETag, gets+appends locally, then `put-object
--if-match <etag>`. Retries up to 100 times on ETag mismatch. Works, but it
downloads and reuploads the *entire* `data.json.gz` on every benchmark that
reports results; the file grows forever.

### Raw data format on S3

- `data.json.gz` - gzipped JSONL. Each line is one measurement record as emitted
  by `vortex-bench::print_measurements_json`. Different measurement types
  produce different shapes (see [`03-raw-data-schema.md`](./03-raw-data-schema.md)).
- `commits.json` - plain JSONL. Each line is a `{author, committer, id, message,
  timestamp, tree_id, url}` record produced by `scripts/commit-json.sh` from
  `git log -1`.
- `file-sizes-*.json.gz` - per-dataset compressed file size measurements,
  appended by the SQL benchmarks workflow.

### Website side: `benchmarks-website/`

- **Backend** (`server.js`, ~600 LOC Node): an HTTP server that on startup and
  every 5 minutes does:
  1. Fetches `commits.json` (or uses local sample).
  2. Streams `data.json.gz` line-by-line, ungzipping on the fly.
  3. For each record, uses `getGroup(b)` to classify it into one of:
     `Random Access`, `Compression`, `Compression Size`, `TPC-H (NVMe) (SF=10)`,
     `Clickbench`, etc. The classifier is a stack of substring checks + regex
     matches against `b.name`, with a `QUERY_SUITES` config table in
     `src/config.js` that's imported directly by the server.
  4. Splits `b.name` on `/` to extract chart and series names, then applies
     `ENGINE_RENAMES` and unit conversions (ns→ms, bytes→MiB).
  5. Builds a nested Map: `group → chart → series → array-aligned-with-commits`.
  6. Pre-computes 1x/2x/4x/8x LTTB downsamples.
  7. Builds a `metadata` object with per-group summary stats (geomean of ratios,
     performance rankings, etc. - see `calcSummary`).

  Exposes:
  - `GET /api/metadata` - all group/chart/series metadata + summaries.
  - `GET /api/data/:group/:chart?last=N&startIdx=&endIdx=&start=&end=` -
    returns one chart's data, auto-selecting downsample level.

- **Frontend** (`src/App.jsx` + 6 components, Vite-built SPA): React + Chart.js
  + chartjs-plugin-zoom. Fetches metadata once, renders collapsible benchmark
  sections, fetches chart data lazily as sections expand. Has engine filters,
  category tags, search, full-screen modal, hash-based deep links.

- **Deploy**: Dockerfile + docker-compose.yml running on an EC2 instance;
  Watchtower polls ghcr.io for new images every 60s.

### The config.js problem

`benchmarks-website/src/config.js` is the "single source of truth" for
displaying benchmark data, and it's duplicated between server and client. It
contains:

- `QUERY_SUITES` - which query suites exist, their prefixes, their fan-out
  (storage x scale factor) dimensions.
- `FAN_OUT_GROUPS` - pre-enumerated list of (suite, storage, SF) triplets.
- `ENGINE_RENAMES` - post-hoc renaming of engine:format strings.
- `BESPOKE_CONFIGS` - hidden/renamed datasets per group, kept-charts lists.
- `CHART_NAME_MAP` - pretty names for chart headers.
- `CATEGORY_TAGS` - sidebar filtering.
- `SERIES_COLOR_MAP`, `FALLBACK_PALETTE`, `SCALE_FACTOR_DESCRIPTIONS`,
  `BENCHMARK_DESCRIPTIONS`, `ENGINE_LABELS`.

Anything that references an engine or a dataset name has to be added to the
right subset of these tables. This is one of the two biggest complaints about
the current site (the other is the append-JSONL storage model).

## What works well today

- SSR means initial load is fast; the React app is purely interactive on top of
  pre-computed metadata.
- The chart UI is OK. `BenchmarkSection`, `ChartContainer`, `Modal`, sidebar
  filters work. They are not glamorous but they work.
- LTTB downsampling produces visually clean charts for long series without
  overwhelming the client.
- The periodic refresh (5min) is cheap enough that we don't need a push
  mechanism.

## What we want to keep in v3 (even if reimplemented)

- The **URL surface** (`/api/metadata`, `/api/data/:group/:chart?last=N`) is
  reasonable. v3 does not need to keep it byte-identical, but the
  "metadata-first, lazy per-chart data fetches" pattern is worth keeping.
- **LTTB pre-downsampling at multiple levels**. At 1000s of data points per
  series, this is the difference between "chart renders" and "tab locks".
- The **summary cards** (geomean ratio vs parquet, random-access rankings,
  per-query geomean score). Users actually look at these.
- **Deep links** to groups (`#group-TPC-H-(NVMe)-(SF=10)` etc.).
- Engine/category filters in the sidebar.

## What we want to change in v3

1. The classification logic (`getGroup`, `formatQuery`, `normalizeChartName`,
   name-splitting) moves into the **ingestion** step, not the viewer. The
   database stores structured columns, not a parseable `name` string.
2. `config.js` gets replaced by **a small set of dimension/lookup tables in
   DuckDB** plus a skinny frontend config for *purely presentational* things
   (colors, friendly labels).
3. Raw storage becomes **a DuckDB file** (or its columnar on-disk
   representation) instead of `data.json.gz`. New runs append rows to a
   `measurements` table instead of concatenating JSON lines.
4. The viewer becomes **Leptos SSR** with interactive hydration only where
   needed (Chart.js can still be used for the actual drawing).

## v2 files worth re-reading before v3 work

- `benchmarks-website/server.js` - for the classifier heuristics we need to
  encode into the ingester.
- `benchmarks-website/src/config.js` - for the set of known groups, engines,
  and rename rules.
- `benchmarks-website/src/components/BenchmarkSection.jsx` - for the summary
  card shapes.
- `vortex-bench/src/measurements.rs` - for the JSON emission side.
- `scripts/cat-s3.sh` + `scripts/commit-json.sh` - for the current append
  mechanism.
