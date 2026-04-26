<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Component: Server (alpha)

## Required reading

- [`../00-overview.md`](../00-overview.md)
- [`../01-schema.md`](../01-schema.md)
- [`../02-contracts.md`](../02-contracts.md)

## Goal

A single Rust binary: an HTTP server that owns a DuckDB file on
local disk, accepts authenticated `/api/ingest` POSTs, and serves
enough of a read API to render one chart page.

This is the **alpha** version. It runs locally or on a dev box; no
production deploy. Production deploy, backups, admin tooling, and
historical data import are deferred (see
[`../deferred.md`](../deferred.md)).

The server crate is `vortex-bench-server`, living at
`benchmarks-website/server/`, registered as a workspace member.

## In scope

- Open the DuckDB file and apply the schema DDL on boot. No
  migration framework yet - if the schema changes during alpha,
  delete the file and re-run.
- Bearer-token middleware on `/api/ingest`. Token from
  `INGEST_BEARER_TOKEN` env var, constant-time compared.
- `POST /api/ingest`: parse the envelope from
  [`../02-contracts.md`](../02-contracts.md), upsert the commit,
  dispatch each record to its destination fact table by `kind`,
  enforce all-or-nothing per POST. Compute each row's
  `measurement_id` server-side as part of the INSERT. Return
  `{ inserted, updated }` aggregated across tables.
- `GET /api/groups` and `GET /api/chart/:slug`: enough to render
  one chart page. Slugs round-trip; the agent picks the format.
- `GET /health`: enough to confirm the DB is open and ingest is
  working (path, latest commit timestamp, per-table row counts -
  exact shape is the agent's call).
- Mount whatever HTML routes the web-ui component contributes.

Framework, templating engine (`maud` or `askama`), DuckDB driver
version, module layout, and DB-access concurrency model are the
agent's call. Pin the DuckDB crate version in `Cargo.toml`.

## Out of scope (deferred)

Schema migrations, lookup tables, pre-built views, multi-page read
API, admin endpoints, containerization, EBS mount, backups. See
[`../deferred.md`](../deferred.md).

## Acceptance criteria

- `cargo build` succeeds for the server crate.
- Integration test: POST a fixture envelope with a valid bearer →
  200; POST again → 200 with `updated > 0, inserted = 0`; POST
  with no/wrong bearer → 401; POST with an unknown `kind` → 400.
- `GET /health` returns a coherent shape after an ingest.
- `cargo run` for the server, pointed at a fresh DuckDB file,
  serves both read routes locally.

## Branch

`claude/benchmarks-v3-server`
