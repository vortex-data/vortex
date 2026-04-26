<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 00 - Overview

## What we're building

A replacement for the current `bench.vortex.dev` site. The new
stack is a **single Rust binary** (axum + maud + duckdb-rs) that
owns a **DuckDB database** on local disk and serves the website
plus an `/api/ingest` route. CI eventually POSTs new benchmark
results there. There is no separate ingester service, no S3
coordination layer for writes, no client-side WASM.

The server crate is `vortex-bench-server` at
`benchmarks-website/server/`.

## Phasing

We build this in two phases. **Plan only the first.**

### Alpha (this plan)

The smallest end-to-end loop that proves the design:

1. **Schema** locked enough to ingest one benchmark result.
2. **Server**: open DuckDB, accept a bearer-token-authenticated POST,
   serve a couple of read routes.
3. **Emitter**: `vortex-bench --gh-json-v3` + a tiny POST script.
4. **Web UI**: one landing page + one chart page rendered against a
   fixture DB.

That's it. No production deploy, no historical data import, no CI
workflow integration, no admin tooling, no schema migration
framework, no auth beyond the shared bearer token. All of those
live in [`deferred.md`](./deferred.md).

The alpha runs on a developer machine. v2 keeps running in
production unchanged. There is no cutover in alpha.

### Phase 2 and beyond

Once the alpha loop is green, we layer in production deploy,
historical migration, CI dual-write, and the rest of the v2-parity
work. Stubs are in [`deferred.md`](./deferred.md).

## Architecture (alpha)

One process, one DB file. The server is the API and the website.
The emitter writes JSONL of bare records; a small POST script
wraps and uploads them. CI isn't wired up yet; ingest happens
manually during alpha.

## Components

Three components for alpha. Each is one workstream, one branch, one
PR.

| Component | Plan | Owns |
|---|---|---|
| Server | [components/server.md](./components/server.md) | DuckDB open + schema, bearer-auth ingest, read routes, HTML routes mounted from web-ui |
| Emitter | [components/emitter.md](./components/emitter.md) | `vortex-bench --gh-json-v3` + the post-ingest script |
| Web UI | [components/web-ui.md](./components/web-ui.md) | Landing page + chart page, against a fixture DuckDB |

### Dependencies

The schema feeds all three components. The contracts feed the
server and the emitter. With both stable, **all three components
can be worked on in parallel**.

## Goals

In priority order:

1. **End-to-end alpha loop works.** Emit → POST → store → render.
2. **Schema is the right shape.** Five fact tables (one per
   measurement family) plus a `commits` dim. See
   [`01-schema.md`](./01-schema.md).
3. **Each component is small enough that one agent can finish it
   in one PR.** No mega-PRs.

Cutover, parity, and "faster than v2" are explicit non-goals at
alpha; they come back in phase 2.

## Shared docs

- [`00-overview.md`](./00-overview.md) (this file)
- [`01-schema.md`](./01-schema.md) - the five fact tables + `commits`
- [`02-contracts.md`](./02-contracts.md) - wire shapes + HTTP error
  matrix + auth header
- [`benchmark-mapping.md`](./benchmark-mapping.md) - existing
  benchmarks → fact tables
- [`decisions.md`](./decisions.md) - resolved decisions
- [`deferred.md`](./deferred.md) - phase-2 stubs

## Status of v2 during alpha

v2 stays in production untouched. Do not edit
`benchmarks-website/server.js`, `benchmarks-website/src/`, or any
other v2 files at `benchmarks-website/` top level. v3 lives in the
sibling subdirectory at `benchmarks-website/server/`
(`vortex-bench-server` crate).
