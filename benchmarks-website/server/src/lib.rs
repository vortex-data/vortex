// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `vortex-bench-server` — the v3 [`bench.vortex.dev`](https://bench.vortex.dev)
//! server.
//!
//! A single Rust binary that owns one DuckDB file on local disk, accepts
//! bearer-authenticated `/api/ingest` POSTs, and serves the JSON read API
//! plus every HTML page. All static assets (Chart.js + the zoom plugin +
//! `chart-init.js` + `style.css` + the two logos) are baked into the
//! binary via `include_bytes!` so a deploy is one binary plus a database
//! file.
//!
//! The hot website path is a materialized latest-100 read model. DuckDB
//! remains the source of truth, but default group/chart hydration is served
//! from precomputed, precompressed in-memory artifacts. See
//! `benchmarks-website/server/ARCHITECTURE.md` for the higher-level model.
//!
//! ## Routes
//!
//! - `GET /` — landing page, every group rendered as chart shells plus
//!   versioned shard metadata. Groups hydrate on intent/open.
//! - `GET /chart/{slug}` — single-chart permalink. Defaults to the
//!   materialized latest-100 window; `?n=all` uses the DB fallback.
//! - `GET /group/{slug}` — every chart shell in one group on a single page,
//!   opened by default and hydrated through the shard path.
//! - `GET /static/...` — the bundled JS / CSS / logos.
//! - `GET /api/groups` — flat list of every group with chart-link metadata.
//!   Defaults to a materialized artifact.
//! - `GET /api/chart/{slug}` — one chart's payload (`history`, `commits`,
//!   `series`, `unit_kind`, ...). Defaults to a materialized latest-100
//!   artifact; `?n=all` and non-default windows use the DB fallback.
//! - `GET /api/group/{slug}` — every chart in one group. Defaults to a
//!   materialized latest-100 compatibility artifact.
//! - `GET /api/artifacts/{generation}/groups/{group_slug}/shards/{index}` —
//!   immutable versioned latest-100 group shard artifact for page hydration.
//! - `GET /health` — liveness probe + per-table row counts.
//! - `POST /api/ingest` — bearer-gated ingest. See [`ingest`] for the HTTP
//!   matrix and [`auth`] for the bearer middleware.
//!
//! ## Module map
//!
//! | Module        | Role                                                                                        |
//! |---------------|---------------------------------------------------------------------------------------------|
//! | [`app`]       | [`app::AppState`] (DB handle + bearer + read store + paths) and the Axum router composition. |
//! | [`auth`]      | Bearer-token middleware for `/api/ingest`.                                                  |
//! | [`db`]        | [`db::DbHandle`] task-local connection cloning + read backpressure + hash helpers.          |
//! | [`schema`]    | DuckDB DDL ([`schema::SCHEMA_DDL`]) and the wire schema version.                            |
//! | [`records`]   | Wire shapes for `POST /api/ingest`.                                                         |
//! | [`ingest`]    | `POST /api/ingest` handler, cache invalidation, and read-model rebuild scheduling.          |
//! | [`error`]     | [`error::IngestError`] and [`error::ApiError`] with their HTTP-status mapping.              |
//! | [`slug`]      | [`slug::ChartKey`] / [`slug::GroupKey`] enums + base64url round-trip.                       |
//! | [`api`]       | Read API. `mod.rs` mounts the handlers; submodules are listed on its module doc.            |
//! | [`html`]      | HTML pages — `mod.rs` mounts the routes; submodules render the body.                        |
//! | [`read_model`]| Materialized latest-100 generations and encoded artifact serving.                           |
//! | [`query_cache`]| Single-flight cache for non-materialized fallback reads.                                    |
//!
//! ## Request flow
//!
//! 1. Axum receives the request and routes by method + path.
//! 2. `/api/ingest` first passes through [`auth::require_bearer`]; other
//!    routes skip auth.
//! 3. The handler parses body / path / query into typed inputs (e.g.
//!    [`slug::ChartKey::from_slug`]).
//! 4. For default latest-100 API reads, the handler serves an encoded
//!    [`read_model`] artifact directly from memory.
//! 5. For ingest, `?n=all`, and non-default windows, the handler hands a
//!    closure to [`db::run_blocking`], which clones a task-local DuckDB
//!    connection and runs synchronous work on `tokio::task::spawn_blocking`.
//! 6. Fallback chart/group reads go through [`query_cache`] so concurrent
//!    identical misses share one compute.
//! 7. Errors are mapped into [`error::IngestError`] / [`error::ApiError`]
//!    with the right HTTP status. HTML responses are rendered via `maud`;
//!    hot JSON artifacts are returned as already encoded bytes.

pub mod api;
pub mod app;
pub mod auth;
pub mod db;
pub mod error;
pub mod html;
pub mod ingest;
pub mod query_cache;
pub mod read_model;
pub mod records;
pub mod schema;
pub mod slug;
