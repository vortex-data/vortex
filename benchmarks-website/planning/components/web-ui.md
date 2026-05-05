<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Component: Web UI (alpha)

## Required reading

- [`../00-overview.md`](../00-overview.md)
- [`../01-schema.md`](../01-schema.md)
- [`../02-contracts.md`](../02-contracts.md) - the JSON shapes you
  render against.

## Goal

Get something on screen. **One landing page** that lists groups and
**one chart page** that renders a single chart. SSR HTML + a thin
Chart.js hydration. That's it for alpha.

This component develops in parallel against a fixture-populated
DuckDB - no dependency on the live ingest path.

## In scope

- A fixture: a small DuckDB file (or a builder that produces one
  from a JSONL fixture) covering all five fact tables with a
  handful of records each. Used for dev and tests.
- Landing page (`GET /`): list of groups with links into chart
  pages, derived from `/api/groups`.
- Chart page (`GET /chart/:slug`): one Chart.js line chart, data
  embedded inline as a JSON `<script>` tag (no client-side
  round-trip after page load).
- Plain CSS. No client-side framework.

Templating engine, exact module layout, fixture format, and any
helper crates are the agent's call. If the server crate already
chose `maud` vs `askama`, follow it.

## Out of scope (deferred)

- Per-commit page, full group landing with filters / modal /
  zoom-pan, ad-hoc SQL page, mobile redesign.
- Engine + category filters, search, full-screen modal, deep links.
- LTTB downsampling.
- Lookup-table-driven engine names and color palettes (use the raw
  `engine:format` strings and a small fallback palette).
- Summary cards (geomean ratios, rankings).

See [`../deferred.md`](../deferred.md).

## Acceptance criteria

- Both routes render against the fixture DB.
- The chart hydrates without a network round-trip after page load.
- Snapshot test of the rendered HTML for both pages, against the
  fixture.
- Manually verified in a real browser; recorded in PR description.

## Branch

`claude/benchmarks-v3-web-ui`
