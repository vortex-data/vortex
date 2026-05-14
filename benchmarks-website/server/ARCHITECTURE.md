<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Benchmark Server Architecture

The benchmark website is optimized around a materialized latest-100 read path.
DuckDB is the source of truth, but normal landing-page and group-open
hydration does not run SQL, serialize JSON, or compress responses per request.

## Hot Read Path

On startup the server builds a `ReadGeneration` from one DuckDB snapshot. That
generation contains precomputed JSON artifacts for:

- `/api/groups`
- default `/api/chart/{slug}` latest-100 payloads
- default `/api/group/{slug}` latest-100 compatibility payloads
- versioned group shard payloads under
  `/api/artifacts/{generation}/groups/{group_slug}/shards/{index}`

Each artifact is stored in memory as identity, gzip, and brotli bytes. Request
handlers negotiate `Accept-Encoding` and serve those bytes directly with
`ETag`, `Vary: Accept-Encoding`, `Content-Length`, and cache headers.

## Page Hydration

The landing page and `/group/{slug}` render group metadata plus chart shells,
not inline chart payloads. Each group carries the active read generation, shard
count, and shard URL prefix. `chart-init.js` fetches shard 0 on intent or group
open so charts paint quickly, then queues the remaining latest-100 shards with
bounded per-tab concurrency.

Latest-100 chart payloads include additive `history` metadata:

- `total_commits`: full x-axis length for the chart
- `start_index`: where this payload starts in the full x-axis
- `loaded_commits`: number of loaded commits
- `complete`: whether the payload covers the full x-axis

The client normalizes incomplete latest-100 payloads onto the full virtual
x-axis. Older unloaded commits are represented by blank labels and null series
values, so the range strip, zoom limits, and slider bounds behave as if the
whole history is present without fabricating data.

## Full-History Warmup

Opening a group queues `/api/chart/{slug}?n=all` for that group's charts in a
separate low-concurrency priority queue. A later-opened group gets higher
priority than queued work for older groups. If the user pans or zooms into an
unloaded virtual range before warmup finishes, that chart's queued full-history
request is promoted. In-flight requests are not cancelled.

When the full payload arrives, the client replaces the virtual latest-100
payload in place and preserves the current x-range when possible.

## Fallback Paths

`?n=all` and non-default `?n=` windows still use the DB-backed fallback path.
Those reads go through `QueryCache` single-flight entries and the DB read
semaphore so cold or unusual requests do not stampede DuckDB. Ingest writes do
not consume read permits.

## Ingest And Rebuild

Successful ingest invalidates `QueryCache` and schedules a read-model rebuild.
The active generation remains live while rebuilding. Repeated rebuild requests
coalesce, and a failed rebuild keeps serving the old generation. The server
keeps the active generation plus the most recent previous generation so already
loaded pages can continue resolving immutable shard URLs across a swap.

## Main Files

- `src/read_model.rs`: materialized generation and encoded artifact serving
- `src/api/mod.rs`: API routing between materialized artifacts and fallbacks
- `src/api/charts.rs`: chart DTO construction and `history` metadata
- `src/html/mod.rs`, `src/html/landing.rs`: shell/shard HTML rendering
- `static/chart-init.js`: virtual-axis normalization, shard hydration, and
  full-history priority warmup
- `src/query_cache.rs`: single-flight fallback cache
- `src/db.rs`: DuckDB connection cloning and read backpressure
