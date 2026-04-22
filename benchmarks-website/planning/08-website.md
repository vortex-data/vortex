<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 08 - Website UX

This doc is the **principles + page inventory** for the v3 site, not a
component-level design. Designers/implementers later can make it pretty.

## Principles

1. **SSR first, hydrate where needed.** The initial HTML of every page is
   complete - no "loading..." spinners for content the server already knows.
   Charts hydrate on the client because Chart.js needs a canvas.

2. **URLs are navigable.** Every meaningful state (selected group, selected
   chart, commit range) maps to a URL. v2 gets this right with
   `#group-<name>`; v3 should do the same with real paths.

3. **Find-the-info-you-need is the primary UX.** The v2 site is a scrolling
   wall. v3 should have:
   - A landing overview with "what's the headline summary".
   - A sidebar/search to jump to a specific benchmark group.
   - A per-commit page linkable from GitHub PRs.

4. **Fewer moving parts per page.** Chart filters should be obvious. Engine
   filters, storage filters, and scale-factor filters should all be in one
   place with consistent affordances.

5. **No "magic" pretty names.** Use the `display_name` from `known_engines` /
   `known_formats` / `known_datasets` tables (see [`05-schema.md`](./05-schema.md)),
   not a hardcoded rename map in the frontend.

## Pages

### `/` - Overview

- Summary card per benchmark group (the stuff `calcSummary` produces in v2):
  - For "Random Access": ranking of engines by latency.
  - For "Compression": geomean compress/decompress throughput vs parquet.
  - For "Compression Size": min/mean/max size ratio vs parquet.
  - For each SQL suite: per-engine geomean score of query-time ratio to
    fastest.
- Each summary card links to `/group/:slug`.
- A "last updated" indicator + a "latest commit" link.

### `/group/:slug` - One benchmark group

- Header with the group's description and category tags.
- Engine filter (one row of toggle buttons).
- Time range selector (last N commits, or date range). Default = last 100.
- Grid or list of charts. View mode toggles.
- Each chart has a full-screen modal (keep the v2 Modal, it works).
- Deep link to each chart via `#chart-<slug>`.

### `/chart/:slug` - Full-screen single chart

Addressable, shareable, embedded-in-PR-friendly. Same chart as the modal but
its own route so it can be linked to.

### `/commit/:sha` - Per-commit snapshot

Given a SHA, show every benchmark's value at that commit (and optionally the
delta vs the previous or parent commit). This is the "was my PR a regression"
page. Link to it from the header + from the commit-message list.

### `/api/...`

Compatibility routes for anything that was scripted against v2's `/api/metadata`
and `/api/data/:group/:chart`. We do not need to guarantee the JSON shape is
identical, but we should keep the endpoints live so nothing external breaks.

### `/health`

Returns the `bench.duckdb` version (ETag or last-modified), row count,
commit count, most recent commit's timestamp. Used by uptime checks and for
debugging "is my merge showing up yet".

## Interactive features

### Must-have

- Pan / zoom on charts (keep chartjs-plugin-zoom; it's fine).
- Pre-downsampled 1x/2x/4x/8x levels per chart (either computed in SQL at
  request time or cached server-side). The client picks a level based on the
  current zoom.
- Engine filter that greys out series rather than removes them, so the axis
  doesn't rescale.
- Copy-link buttons on every chart and group header.

### Nice-to-have (don't block cutover)

- "Compare commits" overlay: pick two commits, highlight the delta on every
  chart.
- Ad-hoc SQL query page (behind a "power users" toggle).
- Export to CSV.

## What we explicitly DON'T want to rebuild

- The `BESPOKE_CONFIGS.renamedDatasets` pattern. Use the `known_engines` /
  `known_formats` / `known_datasets` lookup tables in DuckDB.
- The `CHART_NAME_MAP` that re-pretties chart headers. A chart's display name
  is derived from (metric_kind, dataset, format) at render time with a small
  Rust helper. Overrides, if needed, can live in a data-only
  `group_display_labels` table keyed by a typed enum discriminator.
- Fan-out-group enumeration. Don't pre-list `TPC-H (NVMe) (SF=1)` etc.; list
  them dynamically from the DB.
- `downsample` in Node. Move it server-side or into SQL.

## Technology choices (concrete)

- **Leptos** for the web framework. Default SSR mode.
- **axum** under Leptos for the HTTP router (that's Leptos's current default
  integration).
- **`duckdb` Rust crate** (or `duckdb-rs`) for the DB handle. Open read-only,
  share across request handlers, rebuild the handle on DB refresh.
- **Chart.js** for the actual chart drawing, hydrated on the client. Data for
  each chart is embedded in the server-rendered HTML as a JSON script tag.
- **No build-time CSS framework** to start. Plain CSS or a tiny CSS-in-Rust
  library is fine. Tailwind is optional; we can defer that choice.

## Performance budget

Back-of-envelope targets:

- First byte: <200ms at P50.
- Fully rendered `/` (no client hydration yet): <500ms at P50.
- Interactive (charts drawn): <1.5s at P50.
- DuckDB cold open on server start: <1s for our DB size.

If any of these balloon, the first place to look is "are we pulling all
measurements for every page". The landing page should never read the full
`measurements` table; it should read only the "latest per group" view.
